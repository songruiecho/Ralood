# 用于对LLM生成的结果与原样本数据集进行可视化，判断：两个数据集的相似性以及生成样本本身的多样性
import pandas as pd
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # 用于降维以可视化
import numpy as np
import cfg
import matplotlib.lines as mlines
import json
import ICL
import seaborn as sns  #习惯上简写成sns

def scatter(args, target='sst5'):
    frame = pd.read_csv('datasets/{}/{}/test.tsv'.format(args.task, target), sep='	').dropna()
    if args.task in ['SA', 'TD']:
        texts = list(frame['Text'].values)
    elif args.task in ['NLI']:
        texts = 'Premise: ' + frame['Premise'].astype(str) + ' Hypothesis: ' + frame['Hypothesis'].astype(str)
        texts = list(texts.to_numpy())
    texts = texts[:1000]

    with open('datasets/{}/api_generate_{}.txt'.format(args.task, target), 'r') as rf:
        generate_texts = [each.strip() for each in rf.readlines() if each.strip() != ''][:1000]

    frame = pd.read_csv('datasets/{}/{}/train.tsv'.format(args.task, args.source_datasets[args.task]), sep='	').dropna()
    if args.task in ['SA', 'TD']:
        source_texts = list(frame['Text'].values)[:1000]
    elif args.task in ['NLI']:
        source_texts = 'Premise: ' + frame['Premise'].astype(str) + ' Hypothesis: ' + frame['Hypothesis'].astype(str)
        source_texts = list(source_texts.to_numpy())[:1000]

    scorer = AutoModel.from_pretrained(args.LLM_root + 'SimCSE').cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.LLM_root + 'SimCSE')

    labels = [0] * len(texts) + [1]*len(generate_texts) + [2]*len(source_texts)

    texts = texts + generate_texts + source_texts
    print(len(texts))
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

    pca = PCA(n_components=2)  # 降到2维以便于可视化
    X_pca = pca.fit_transform(embeddings)
    random_df = pd.DataFrame({
        'X': X_pca[:, 0],
        'Y': X_pca[:, 1],
        'Label': labels
    })
    random_df['Label'] = random_df['Label'].astype('category')
    plt.figure(dpi=500)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='winter', s=10)
    # plt.xlabel('PCA Feature 1')
    # plt.ylabel('PCA Feature 2')
    # 添加图例
    # 使用类别代码作为颜色索引，但这里我们直接使用类别代码的颜色映射
    colors = plt.cm.winter(np.linspace(0, 1, len(random_df['Label'].cat.categories)))
    random_df['Color'] = random_df['Label'].cat.codes
    # 为每个类别创建一个图例项
    handles = []
    for label, color in zip(np.unique(labels), colors):
        handles.append(plt.scatter([], [], c=[color], label=[target, 'generated {}'.format(target), args.source_datasets[args.task]][label], s=20))

        # 添加图例到图表中
    plt.legend(handles=handles, loc='best')
    plt.show()

def vis_sim_div(args, target='sst5'):
    ''' 对相似性和多样性的结果进行比较
    :return:
    '''
    # args.score_func = 'multi'
    # ICL.topk_retrieve(args, target=target, confidence=0.9)
    # args.score_func = 'simcse'
    # ICL.topk_retrieve(args, target=target, confidence=0.9)
    with open('datasets/{}/api_generate_{}.txt'.format(args.task, target), 'r') as rf:
        generate_texts = [each.strip() for each in rf.readlines() if each.strip() != '']

    frame = pd.read_csv('datasets/{}/{}/test.tsv'.format(args.task, target), sep='	').dropna()
    if args.task in ['SA', 'TD']:
        texts = list(frame['Text'].values)[:5000]
    elif args.task in ['NLI']:
        texts = 'Premise: ' + frame['Premise'].astype(str) + ' Hypothesis: ' + frame['Hypothesis'].astype(str)
        texts = list(texts.to_numpy())[:5000]

    file = open('topk/{}_{}_{}.json'.format(args.task, 'multi', target), 'r', encoding='utf-8')
    div_topk = json.load(file)

    file = open('topk/{}_{}_{}.json'.format(args.task, 'simcse', target), 'r', encoding='utf-8')
    sim_topk = json.load(file)

    scorer = AutoModel.from_pretrained(args.LLM_root + 'SimCSE').cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.LLM_root + 'SimCSE')

    sim_texts = []
    div_texts = []
    current_texts = []
    sim_count, div_count = 0, 0
    for test_sample, context in zip(texts, div_topk):
        current_texts.append(test_sample)
        for label in context.keys():
            for i in range(3):
                try:
                    text = generate_texts[context[label][i]]
                    if text not in div_texts:
                        div_texts.append(text)
                except:
                    continue
        sim_count += 1
        if sim_count == 100:
            break
    for test_sample, context in zip(texts, sim_topk):
        for label in context.keys():
            for i in range(3):
                try:
                    text = generate_texts[context[label][i]]
                    if text not in sim_texts:
                        sim_texts.append(text)
                except:
                    continue
        div_count += 1
        if div_count == 100:
            break

    labels = [0] * len(sim_texts) + [1] * len(div_texts) + [2] * len(current_texts)
    texts = sim_texts + div_texts + current_texts

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

    pca = PCA(n_components=2)  # 降到2维以便于可视化
    X_pca = pca.fit_transform(embeddings)
    random_df = pd.DataFrame({
        'X': X_pca[:, 0],
        'Y': X_pca[:, 1],
        'Label': labels
    })
    random_df['Label'] = random_df['Label'].astype('category')
    plt.figure(dpi=500)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='winter', s=30)
    # plt.xlabel('PCA Feature 1')
    # plt.ylabel('PCA Feature 2')
    # 添加图例
    # 使用类别代码作为颜色索引，但这里我们直接使用类别代码的颜色映射
    colors = plt.cm.winter(np.linspace(0, 1, len(random_df['Label'].cat.categories)))
    random_df['Color'] = random_df['Label'].cat.codes
    # 为每个类别创建一个图例项
    handles = []
    for label, color in zip(np.unique(labels), colors):
        handles.append(plt.scatter([], [], c=[color], label=['sim', 'div', 'test samples'][label], s=20))

        # 添加图例到图表中
    plt.legend(handles=handles, loc='best')
    plt.show()

def confidence_distribution(args, target):

    target_frame = pd.read_csv('datasets/{}/{}/test.tsv'.format(args.task, target), sep='	').dropna()
    if args.task in ['SA', 'TD']:
        target_texts = list(target_frame['Text'].values)[:5000]
        target_labels = list(target_frame['Label'].values)[:5000]
    elif args.task in ['NLI']:
        target_texts = 'Premise: ' + target_frame['Premise'].astype(str) + ' Hypothesis: ' + target_frame['Hypothesis'].astype(str)
        target_texts = list(target_texts.to_numpy())[:5000]
        target_labels = list(target_frame['Label'].values)[:5000]

    text2label = dict(zip(target_texts, target_labels))

    frame = pd.read_csv("retrieve_results/{}.csv".format(target))

    bins = [i / 10 for i in range(11)]  # 生成 0~1 间的区间 [0, 0.1, 0.2, ..., 1.0]
    labels = [f'({bins[i]}~{bins[i + 1]}]' for i in range(len(bins) - 1)]  # 创建区间标签
    # 使用 pd.cut 函数进行分组
    frame['Confidence_group'] = pd.cut(frame['Confidence'], bins=bins, labels=labels, include_lowest=True)
    filtered_df = frame[frame['Text'].isin(target_texts)]
    # 查看结果
    grouped = filtered_df.groupby('Confidence_group')

    for group_name, g_frame in grouped:
        print(group_name)
        g_frame['Predicted_Label'] = g_frame['Text'].map(text2label)
        # 计算预测是否正确
        g_frame['Correct'] = g_frame['Predicted_Label'] == g_frame['Label']

        # 按照 Confidence_group 分组，并计算每组的准确率
        accuracy_by_group = g_frame.groupby('Confidence_group').apply(
            lambda x: x['Correct'].sum() / len(x) if len(x) > 0 else 0
        )

        specific_accuracy = accuracy_by_group.loc[group_name]

        print(g_frame.shape[0], specific_accuracy)

def DKE(args, target):
    with open('datasets/{}/api_generate_{}.txt'.format(args.task, target), 'r') as rf:
        generate_texts = [each.strip() for each in rf.readlines() if each.strip() != '']

    frame = pd.read_csv('datasets/{}/{}/test.tsv'.format(args.task, target), sep='	').dropna()
    if args.task in ['SA', 'TD']:
        texts = list(frame['Text'].values)[:5000]
    elif args.task in ['NLI']:
        texts = 'Premise: ' + frame['Premise'].astype(str) + ' Hypothesis: ' + frame['Hypothesis'].astype(str)
        texts = list(texts.to_numpy())[:5000]

    file = open('topk/{}_{}_{}.json'.format(args.task, 'multi', target), 'r', encoding='utf-8')
    div_topk = json.load(file)

    file = open('topk/{}_{}_{}.json'.format(args.task, 'simcse', target), 'r', encoding='utf-8')
    sim_topk = json.load(file)

    scorer = AutoModel.from_pretrained(args.LLM_root + 'SimCSE').cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.LLM_root + 'SimCSE')

    sim_texts = []
    div_texts = []
    current_texts = []
    sim_count, div_count = 0, 0
    for test_sample, context in zip(texts, div_topk):
        current_texts.append(test_sample)
        for label in context.keys():
            for i in range(3):
                try:
                    text = generate_texts[context[label][i]]
                    if text not in div_texts:
                        div_texts.append(text)
                except:
                    continue
        sim_count += 1
        if sim_count == 100:
            break
    for test_sample, context in zip(texts, sim_topk):
        for label in context.keys():
            for i in range(3):
                try:
                    text = generate_texts[context[label][i]]
                    if text not in sim_texts:
                        sim_texts.append(text)
                except:
                    continue
        div_count += 1
        if div_count == 100:
            break

    # labels = [0] * len(sim_texts) + [1] * len(div_texts) + [2] * len(current_texts)
    texts = sim_texts + div_texts + current_texts

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

    pca = PCA(n_components=2)  # 降到2维以便于可视化
    datas = pca.fit_transform(embeddings)

    # 设置全局字体为Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.figure(figsize=(1.5, 1.5), dpi=600)

    sns.kdeplot(x=datas[:len(sim_texts), 0], y=datas[:len(sim_texts), 1], cmap="Blues", fill=True, alpha=0.99, label="sim")
    sns.kdeplot(x=datas[len(sim_texts):len(sim_texts)+len(div_texts), 0], y=datas[len(sim_texts):len(sim_texts)+len(div_texts), 1], cmap="Reds", fill=True, alpha=0.7, label='div', linestyle='--')
    sns.kdeplot(x=datas[len(sim_texts)+len(div_texts):, 0], y=datas[len(sim_texts)+len(div_texts):, 1], cmap="Greens", fill=True, alpha=0.8, label='test', linestyle='--')

    # 自定义图例项
    legend_elements = [
        plt.Line2D([0], [0], color='blue', lw=3, label='sim'),
        plt.Line2D([0], [0], color='red', lw=3, label='div'),
        plt.Line2D([0], [0], color='green', lw=3, label='text')
    ]
    plt.legend(handles=legend_elements, loc='upper right',
               fontsize='xx-small',  # 设置字体大小
               handlelength=0.2,  # 设置图例标记的长度
               handletextpad=0.1)  # 设置图例标记和文本之间的距离
    # plt.gca().patch.set_edgecolor('none')  # 移除边框
    plt.gca().axis('off')
    # 添加图例
    # plt.legend(handles=legend_elements)
    # plt.legend(['Amazon', testset])
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False,
                    labelleft=False)
    # plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    # 显示图形
    plt.show()



if __name__ == '__main__':
    args = cfg.Config()
    target = 'toxigen'
    args.task = 'TD'
    args.batch = 128
    # scatter(args, target)
    # vis_sim_div(args, 'sst5')
    # DKE(args, 'anli')
    # args.task = 'SA'
    # DKE(args, 'dynasent')
    # args.task = 'TD'
    # DKE(args, 'adv_civil')
    # DKE(args, 'semeval')
    # DKE(args, 'sst5')
    # 计算confidence不同区间在目标域上预测的数量以及准确率
    # confidence_distribution(args, 'adv_civil')
    # confidence_distribution(args, 'implicit_hate')
    # confidence_distribution(args, 'toxigen')

    with open('datasets/{}/api_generate_{}.txt'.format(args.task, target), 'r') as rf:
        datas = [each.strip() for each in rf.readlines() if len(each) > 10]

    print(len(datas))

# wanli 4992/10000
# contract_nli  596/10000
# anli  2483/10000

# sst5 7944/10000
# dynasent 7933/10000
# semeval 7792/10000

# adv_civil 5765/10000
# implicit_hate  7807/10000
# toxige  4307/10000