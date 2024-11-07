# The scoring function needed for refined retrieval can be replaced with any defined method
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def simcse_scorer(args, texts, labels, target_texts, topK=10):
    scorer = AutoModel.from_pretrained(args.LLM_root + 'SimCSE').cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.LLM_root + 'SimCSE')
    # 依batch获取表征并转化到cpu上【为了避免维数灾难可以尝试使用PCA进行降维】
    embeddings, t_embeddings = [], []
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

    # 目标域也进行embedding
    for i in tqdm(range(0, len(target_texts), args.batch)):
        batch = target_texts[i:i + args.batch]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
        # 将输入张量移动到CUDA设备上
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.cuda()
        with torch.no_grad():
            outputs = scorer(**inputs).pooler_output
        outputs = outputs.detach().cpu().numpy()
        t_embeddings.append(outputs)
    t_embeddings = np.concatenate(t_embeddings, axis=0)

    # 计算相似度并选择每个样本的topk相似度的样本，并封装成dict
    sim_matrix = cosine_similarity(t_embeddings, embeddings)

    all_sim_records = []
    for i in tqdm(range(sim_matrix.shape[0])):
        sorted_idx = np.argsort(sim_matrix[i, :])[::-1]
        if args.task == 'SA':
            # 遍历sorted_idx从中寻找每个类别最相似（最不像）的样本
            sims = {"negative": [], "positive": [], "neutral": []}
            for idx in sorted_idx:
                if target_texts[i] == texts[idx]:   # 相同样本不能作为结果
                    continue
                if len(sims["negative"]) >= topK and len(sims["positive"]) >= topK and len(sims["neutral"]) >= topK:
                    break
                if len(sims[args.label_space[args.task][labels[idx]]]) < topK:
                    sims[args.label_space[args.task][labels[idx]]].append(idx)
        if args.task == 'TD':
            # 遍历sorted_idx从中寻找每个类别最相似（最不像）的样本
            sims = {"benign": [], "toxic": []}
            for idx in sorted_idx:
                if target_texts[i] == texts[idx]:  # 相同样本不能作为结果
                    continue
                if len(sims["benign"]) >= topK and len(sims["toxic"]) >= topK:
                    break
                if len(sims[args.label_space[args.task][labels[idx]]]) < topK:
                    sims[args.label_space[args.task][labels[idx]]].append(idx)
        if args.task == 'NLI':
            # 遍历sorted_idx从中寻找每个类别最相似（最不像）的样本
            sims = {"entailment":[], 'neutral':[], 'contradiction':[]}
            for idx in sorted_idx:
                if target_texts[i] == texts[idx]:  # 相同样本不能作为结果
                    continue
                if len(sims["entailment"]) >= topK and len(sims["neutral"]) >= topK and len(sims["contradiction"]):
                    break
                if len(sims[args.label_space[args.task][labels[idx]]]) < topK:
                    sims[args.label_space[args.task][labels[idx]]].append(idx)

        all_sim_records.append(sims)
    return all_sim_records

def tfidf_scorer(args, texts, labels, target_texts, topK=10):
    # 将文本转换为TF-IDF向量
    train_datas = [each.strip().replace('Premise: ', '').replace(' Hypothesis: ', '') for each in texts]
    test_datas = [each.strip().replace('Premise: ', '').replace(' Hypothesis: ', '') for each in target_texts]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(train_datas + test_datas)  # 合并两个数组以共享词汇表

    # 分割回原始数组
    X1 = X[:len(train_datas)]
    X2 = X[len(train_datas):]

    # 计算余弦相似度矩阵
    # 注意：这里计算的是text1中每个元素与text2中每个元素的相似度
    sim_matrix = cosine_similarity(X2, X1)

    all_sim_records = []
    for i in tqdm(range(sim_matrix.shape[0])):
        sorted_idx = np.argsort(sim_matrix[i, :])[::-1]
        if args.task == 'SA':
            # 遍历sorted_idx从中寻找每个类别最相似（最不像）的样本
            sims = {"negative": [], "positive": [], "neutral": []}
            for idx in sorted_idx:
                if target_texts[i] == texts[idx]:   # 相同样本不能作为结果
                    continue
                if len(sims["negative"]) >= topK and len(sims["positive"]) >= topK and len(sims["neutral"]) >= topK:
                    break
                if len(sims[args.label_space[args.task][labels[idx]]]) < topK:
                    sims[args.label_space[args.task][labels[idx]]].append(idx)
        if args.task == 'TD':
            # 遍历sorted_idx从中寻找每个类别最相似（最不像）的样本
            sims = {"benign": [], "toxic": []}
            for idx in sorted_idx:
                if target_texts[i] == texts[idx]:  # 相同样本不能作为结果
                    continue
                if len(sims["benign"]) >= topK and len(sims["toxic"]) >= topK:
                    break
                if len(sims[args.label_space[args.task][labels[idx]]]) < topK:
                    sims[args.label_space[args.task][labels[idx]]].append(idx)
        if args.task == 'NLI':
            # 遍历sorted_idx从中寻找每个类别最相似（最不像）的样本
            sims = {"entailment":[], 'neutral':[], 'contradiction':[]}
            for idx in sorted_idx:
                if target_texts[i] == texts[idx]:  # 相同样本不能作为结果
                    continue
                if len(sims["entailment"]) >= topK and len(sims["neutral"]) >= topK and len(sims["contradiction"]):
                    break
                if len(sims[args.label_space[args.task][labels[idx]]]) < topK:
                    sims[args.label_space[args.task][labels[idx]]].append(idx)

        all_sim_records.append(sims)
    return all_sim_records

def random_scorer(args, texts, labels, target_texts, topK=10):
    '''
    :param train_datas:
    :param test_datas:
    :return:  随机生成一个相似度矩阵
    '''
    n, m = len(texts), len(target_texts)
    sim_matrix = np.zeros((m, n), dtype=int)
    # 对每一行，生成0到m-1的随机排列并填充到矩阵中
    for i in range(m):
        # np.random.permutation(m) 生成一个长度为m的数组，包含0到m-1的随机排列
        row = np.random.permutation(n)
        # 将生成的随机排列赋值给矩阵的当前行
        sim_matrix[i, :] = row

    all_sim_records = []
    for i in tqdm(range(sim_matrix.shape[0])):
        sorted_idx = np.argsort(sim_matrix[i, :])[::-1]
        if args.task == 'SA':
            # 遍历sorted_idx从中寻找每个类别最相似（最不像）的样本
            sims = {"negative": [], "positive": [], "neutral": []}
            for idx in sorted_idx:
                if target_texts[i] == texts[idx]:  # 相同样本不能作为结果
                    continue
                if len(sims["negative"]) >= topK and len(sims["positive"]) >= topK and len(sims["neutral"]) >= topK:
                    break
                if len(sims[args.label_space[args.task][labels[idx]]]) < topK:
                    sims[args.label_space[args.task][labels[idx]]].append(idx)
        if args.task == 'TD':
            # 遍历sorted_idx从中寻找每个类别最相似（最不像）的样本
            sims = {"benign": [], "toxic": []}
            for idx in sorted_idx:
                if target_texts[i] == texts[idx]:  # 相同样本不能作为结果
                    continue
                if len(sims["benign"]) >= topK and len(sims["toxic"]) >= topK:
                    break
                if len(sims[args.label_space[args.task][labels[idx]]]) < topK:
                    sims[args.label_space[args.task][labels[idx]]].append(idx)
        if args.task == 'NLI':
            # 遍历sorted_idx从中寻找每个类别最相似（最不像）的样本
            sims = {"entailment":[], 'neutral':[], 'contradiction':[]}
            for idx in sorted_idx:
                if target_texts[i] == texts[idx]:  # 相同样本不能作为结果
                    continue
                if len(sims["entailment"]) >= topK and len(sims["neutral"]) >= topK and len(sims["contradiction"]):
                    break
                if len(sims[args.label_space[args.task][labels[idx]]]) < topK:
                    sims[args.label_space[args.task][labels[idx]]].append(idx)

        all_sim_records.append(sims)
    return all_sim_records

def diversity_std(sim):
    if sim.shape[0] <= 1:
        return 0

    distance_matrix = 1 - np.dot(sim, sim.T)
    std_score = np.std(distance_matrix)
    # 归一化到 0-1 区间
    max_std = 1  # 假设最大值
    min_std = 0  # 最小值
    normalized_std = (std_score - min_std) / (max_std - min_std)
    return normalized_std

def is_diverse(sim, new_sample):
    original_entropy = diversity_std(sim) + 1e-8
    # 加入新样本
    new_entropy = diversity_std(new_sample)
    # 判断多样性是否增加
    return new_entropy/original_entropy > 0.95   # 保证不剧烈下降即可

def judge_full(label2sims, topK):
    '''
    :param label2sims: 类似{"negative": [], "positive": [], "neutral": []}的字典
    :param topK: 每一个元素下的数组中的最大值
    :return: True or False [True表示每一个元素下的数组中的最大值都满足了topK]
    '''
    for key in label2sims.keys():
        if len(label2sims[key]) < topK:
            return False
    return True

def sim_div_scorer(args, texts, labels, target_texts, topK=10):
    all_sim_records = []
    texts, labels, target_texts = texts, np.array(labels), target_texts  # for test
    scorer = AutoModel.from_pretrained(args.LLM_root + 'SimCSE').cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.LLM_root + 'SimCSE')

    embeddings, t_embeddings = [], []
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

    # 目标域也进行embedding
    for i in tqdm(range(0, len(target_texts), args.batch)):
        batch = target_texts[i:i + args.batch]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
        # 将输入张量移动到CUDA设备上
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.cuda()
        with torch.no_grad():
            outputs = scorer(**inputs).pooler_output
        outputs = outputs.detach().cpu().numpy()
        t_embeddings.append(outputs)
    t_embeddings = np.concatenate(t_embeddings, axis=0)

    # 每个待预测样本与待检索样本的相似度
    sim_matrix = cosine_similarity(t_embeddings, embeddings)

    for i in tqdm(range(sim_matrix.shape[0])):
        sorted_idx = np.argsort(sim_matrix[i, :])[::-1]   # 根据与待预测样本相似度进行选择
        # 搜索使得检索结果多样性最大化的新样本，注意检索的过程需要区分不同的label
        if args.task == 'SA':
            # 遍历sorted_idx从中寻找每个类别最相似（最不像）的样本
            label2sims = {"negative": [], "positive": [], "neutral": []}   # 这个保证每个label下找到的样本数量是一定的
        if args.task == 'TD':
            # 遍历sorted_idx从中寻找每个类别最相似（最不像）的样本
            label2sims = {"benign": [], "toxic": []}   # 这个保证每个label下找到的样本数量是一定的
        if args.task == 'NLI':
            label2sims = {"entailment":[], 'neutral':[], 'contradiction':[]}
        # 创建包含标签和对应索引的 DataFrame
        data = pd.DataFrame({
            'label': labels[sorted_idx],
            'index': sorted_idx
        })
        # 根据标签分组，并获取索引
        grouped = data.groupby('label').apply(lambda x: x['index'].tolist())
        # 根据标签以及对应的索引进行后续操作
        for l, l_idx in enumerate(grouped):
            label = args.label_space[args.task][l]   # 先把最相似的放入其中
            label2sims[label].append(l_idx[0])
            for idx in l_idx[1:]:  # 每个idx的多样性查找
                if target_texts[i].lower() == texts[idx].lower():  # 相同样本不能作为结果
                    continue
                concurrent_sims = label2sims[label] + [idx]
                old_embd = [embeddings[label2sims[args.label_space[args.task][lll]]] for lll in list(range(len(grouped)))]
                old_embd, new_embd = np.concatenate(old_embd), embeddings[concurrent_sims]
                if is_diverse(old_embd, new_embd):
                    label2sims[label].append(idx)
                if len(label2sims[label]) >= topK:   # 跳出
                    break
        for l, l_idx in enumerate(grouped):
            label = args.label_space[args.task][l]
            for id in l_idx:
                if id not in label2sims[label] and len(label2sims[label]) < 10:
                    label2sims[label].append(id)
        all_sim_records.append(label2sims)
    return all_sim_records