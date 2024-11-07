# 基于不同目标域的演示从LLMs中生成数据集，首先对样本进行聚类，根据不同的聚落进行代表样本的选择，然后生成符合目标域但是多样性强的文本
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

import cfg
import pandas as pd
import torch
from labeler import *
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import string
import re
import time
from scorer import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # 用于降维以可视化


def remove_usernames_and_links(text):
    # 定义正则表达式模式来匹配用户名和http链接
    pattern = r'@\w+|http\S+'

    # 使用re.sub()函数来替换匹配的部分为空字符串
    cleaned_text = re.sub(pattern, '', text).strip()

    return cleaned_text

def load_LLMs(cfg):
    if 'gpt2' in cfg.LLM:
        tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_path, trust_remote_code=True, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(cfg.LLM_path, trust_remote_code=True)

    if 'llama3' in cfg.LLM.lower():
        tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_path, trust_remote_code=True,
                                                   padding_side='left', )
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(cfg.LLM_path, trust_remote_code=True)

    if 'Qwen' in cfg.LLM:
        tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_path, trust_remote_code=True, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(cfg.LLM_path, trust_remote_code=True)

    if 'gpt-j' in cfg.LLM:
        model = AutoModelForCausalLM.from_pretrained(cfg.LLM_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_path, trust_remote_code=True, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token

    model = model.half().cuda()
    print("(*^_^*) model load finished on {}!!!! ".format(model.device))
    model.eval()
    return model, tokenizer


def cluster(args, target, n_cluster=5):
    frame = pd.read_csv('datasets/{}/{}/test.tsv'.format(args.task, target), sep='	').dropna()
    if args.task in ['SA', 'TD']:
        texts = list(frame['Text'].values)
        labels = list(frame['Label'].values)
    elif args.task in ['NLI']:
        texts = 'Premise: ' + frame['Premise'].astype(str) + ' Hypothesis: ' + frame['Hypothesis'].astype(str)
        texts = list(texts.to_numpy())
        labels = list(frame['Label'].values)

    scorer = AutoModel.from_pretrained(args.LLM_root + 'SimCSE').cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.LLM_root + 'SimCSE')

    embeddings = []
    for i in tqdm(range(0, len(texts), args.batch)):
        batch = texts[i:i + args.batch]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
        # 将输入张量移动到CUDA设备上
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.cuda()
        with torch.no_grad():
            outputs = scorer(**inputs).pooler_output
            outputs = outputs.detach().cpu().numpy()
            embeddings.append(outputs)
    embeddings = np.concatenate(embeddings, axis=0)

    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    kmeans.fit(embeddings)

    # 获取聚类标签
    labels = kmeans.labels_
    plt.figure(dpi=1000)
    # 可视化聚类结果（仅当特征维度较低时有效，这里使用PCA进行降维）
    pca = PCA(n_components=2)  # 降到2维以便于可视化
    X_pca = pca.fit_transform(embeddings)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    # plt.xlabel('PCA Feature 1')
    # plt.ylabel('PCA Feature 2')
    # plt.title('K-means Clustering of Documents (PCA-reduced data)')
    plt.colorbar(label='Cluster Label')
    plt.show()

    # 输出聚类中心和每个文档的聚类标签
    print("Cluster Centers:")
    print(kmeans.cluster_centers_)
    print("\nDocument Labels:")
    print(labels)

    # 创建DataFrame来保存文本ID和标签
    # df = pd.DataFrame({
    #     'text_id': list(range(len(texts))),  # 文本ID
    #     'cluster_label': labels  # 聚类标签
    # })
    #
    # # 将DataFrame保存到CSV文件
    # csv_filename = 'datasets/{}/cluster_{}.csv'.format(args.task, target)
    # df.to_csv(csv_filename, index=False, encoding='utf-8-sig')  # index=False表示不保存行索引
    #
    # print(f"聚类结果已保存到 {csv_filename}")


def sample_generate(args, target, sampled_samles=10000):
    '''
    :param args:
    :param target:
    :param sampled_samles: 一共想要采样的样本数量
    :return:
    '''
    datas, generate_results = [], []
    csv_filename = 'datasets/{}/cluster_{}.csv'.format(args.task, target)
    df = pd.read_csv(csv_filename)
    df_target = pd.read_csv('datasets/{}/{}/test.tsv'.format(args.task, target), sep='	').dropna()
    texts = df_target['Text'].tolist()
    # 创建一个字典来存储每个cluster_label对应的列表
    groups = {label: [] for label in df['cluster_label']}
    # 将数据分配到对应的组中
    for _, row in df.iterrows():
        groups[row['cluster_label']].append(row['text_id'])
    for i in tqdm(range(sampled_samles)):
        data = [random.choice(groups[key]) for key in groups.keys()]
        datas.append(data)
    # 获取样本的prompt
    prompts = []
    for data in datas:
        prompt = 'Please generate a similar text from the following samples:\n'
        prompt += '\n'.join([texts[each] for each in data[:5]])
        prompt = prompt + '\n'
        prompts.append(prompt)

    model, tokenizer = load_LLMs(args)

    for i in tqdm(range(0, len(prompts), args.batch)):
        batch = prompts[i:i+args.batch]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        with torch.no_grad():
            generate_ids = model.generate(input_ids.cuda(), attention_mask=inputs.attention_mask.cuda(),
                                          pad_token_id=0, max_new_tokens=128)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for out, inp in zip(output, batch):
            out = out.replace(inp, '').split('\n')[0]
            if out not in generate_results:
                generate_results.append(out)
            if len(out) > 10:
                out = remove_usernames_and_links(out)
                if out != '':
                    generate_results.append(out)

    with open('datasets/{}/generate_{}.txt'.format(args.task, target), 'w') as wf:
        wf.write('\n'.join(generate_results))




if __name__ == '__main__':
    args = cfg.Config()
    args.task = 'TD'
    target = 'adv_civil'
    args.batch = 128
    cluster(args, target, n_cluster=10)
    # sample_generate(args, target)