# 解耦检索器，将样本域相关和域无关表征分离，使用域相关表征进行域鉴别，域无关表征进行分类
import os

import scorer

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
from os.path import join, exists
import torch.functional as F
import json
import pandas as pd
import cfg
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW, Adam
from torch.autograd import Function
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import random
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KernelDensity
import ICL_baselines

tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/'+'roberta')
# tokenizer = AutoTokenizer.from_pretrained('/home/tianmingjie/songrui/models/roberta_en_base/')

def batch_info_nce_loss(anchor, positive, negatives, temperature=0.1, eps=1e-8):
    """
    计算批处理的 InfoNCE 损失，使用数值稳定性技术防止溢出。

    参数:
    - anchor: (batch_size, embedding_dim) 锚点样本的嵌入
    - positive: (batch_size, embedding_dim) 正样本的嵌入
    - negatives: (batch_size, embedding_dim) 负样本的嵌入
    - temperature: 温度参数，控制相似度的放大程度
    - eps: 非常小的值，用于数值稳定性处理

    返回:
    - loss: 平均批处理的 InfoNCE 损失
    """
    # 计算 anchor 和 positive 的余弦相似度
    positive_similarity = torch.cosine_similarity(anchor, positive, dim=-1)
    positive_similarity = torch.clamp(positive_similarity, min=-2 + eps, max=2 - eps)
    positive_similarity /= temperature
    # 计算 anchor 和 negatives 的余弦相似度
    negative_similarities = torch.matmul(anchor, negatives.transpose(0, 1))
    negative_similarities = torch.clamp(negative_similarities, min=-2 + eps, max=2 - eps)
    negative_similarities /= temperature
    # 稳定地计算正样本的 log-exp
    positive_exp = torch.exp(positive_similarity)
    # 计算负样本的 log-sum-exp
    negative_exp_sum = torch.exp(negative_similarities)
    # 损失是 -log(positive_exp / (positive_exp + sum(negative_exp)))
    loss = -torch.log(positive_exp / (positive_exp + negative_exp_sum))

    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0, reduction='mean', ignore_index=-100):
        """
        :param alpha: 类别权重，默认为 None，如果不为 None，则应为长度等于类别数量的列表或张量。
        :param gamma: 聚焦因子，gamma 越大，对难分类样本的关注越多。
        :param reduction: 损失的聚合方式，'mean' 表示取均值，'sum' 表示求和，'none' 表示不聚合。
        :param ignore_index: -100表示该样本的标签损失并不计算
        """
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (list, torch.Tensor)):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = self.ce(inputs, targets)

        # 获取每个样本的概率，取对数概率的反面
        p_t = torch.exp(-ce_loss)

        # 根据 targets 索引得到对应类别的 alpha 值
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
        else:
            alpha_t = 1.0

        # Focal Loss 公式
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        # 根据 reduction 参数选择返回值
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class Labler(nn.Module):
    def __init__(self, model, num_classes=3):
        super(Labler, self).__init__()
        # BERT Backbone
        self.model = model
        hidden_size = 768

        # 分类器
        self.classifier = nn.Linear(hidden_size, num_classes)  # 用于类别分类的输出

    def forward(self, ipt):
        input_ids = ipt['input_ids']
        attention_mask = ipt['attention_mask']
        outputs = self.model(input_ids=input_ids,
                            attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]

        class_logits = self.classifier(pooled_output)

        return class_logits, pooled_output

    def save(self, model, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model, save_path)

# 创建 PyTorch Dataset
class TextDataset(Dataset):
    def __init__(self, tokenizer, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx].lower()
        label = self.labels[idx]
        return text, label

def collate_fn(batch):
    # 检查 batch 中的样本是 (encoding, label) 还是只有 encoding
    texts = [item[0] for item in batch]  # 获取编码后的文本
    labels = [item[1] for item in batch]
    # 用 tokenizer 来对 batch 进行 padding
    batch_encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors='pt',  # 返回 PyTorch 的 Tensor 格式
        max_length=128  # 设置最大长度以防止 excessive nesting
    )

    batch_encoding['labels'] = torch.tensor(labels)

    return batch_encoding


def train_labeler(args, source):
    # 加载数据集
    df_source = pd.read_csv('datasets/{}/{}/train.tsv'.format(args.task, source), sep='	').dropna()
    df_target = pd.read_csv('datasets/{}/{}/test.tsv'.format(args.task, source), sep='	').dropna()

    if args.task in ['SA', 'TD']:
        texts_source = df_source['Text'].tolist()
        labels_source = df_source['Label'].tolist()
        texts_target = df_target['Text'].tolist()
        labels_target = df_target['Label'].tolist()

    elif args.task in ['NLI']:
        texts_source = 'Premise: ' + df_source['Premise'].astype(str) + ' Hypothesis: ' + df_source['Hypothesis'].astype(str)
        texts_source = list(texts_source.to_numpy())
        labels_source = df_source['Label'].tolist()
        texts_target = 'Premise: ' + df_target['Premise'].astype(str) + ' Hypothesis: ' + df_target['Hypothesis'].astype(str)
        labels_target = list(df_target['Label'].to_numpy())

    train_dataset = TextDataset(tokenizer, texts_source, labels_source)
    val_dataset = TextDataset(tokenizer, texts_target, labels_target)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

    # 加载 RoBERTa tokenizer 和模型
    model = AutoModel.from_pretrained(args.LLM_root + 'roberta')

    da_model = Labler(model, args.nclass)
    da_model = torch.nn.DataParallel(da_model, [0])
    da_model.cuda()

    # 优化器
    optimizer = AdamW(da_model.parameters(), lr=1e-5)

    ce_loss = torch.nn.CrossEntropyLoss()
    # 执行训练过程
    metrics = 0.0
    for epoch in range(3):
        da_model.train()

        for step, batch in enumerate(train_dataloader):
            ipt = {k: v.cuda() for k, v in batch.items()}
            class_logits, fea = da_model(ipt)
            loss = ce_loss(class_logits, ipt['labels'].cuda())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            if step % 50 == 0:
                print("epoch-{}, step-{}/{}, loss:{}".format(epoch, step, len(train_dataloader), loss.data))

        # 模型测试
        acc = dev_model(da_model, dev_loader=val_dataloader)
        if acc > metrics:
            save_path = 'retriever/{}_labeler.pth'.format(source)
            torch.save(da_model.module, save_path)
            metrics = acc
        print('=======================current acc: {}, best acc:{}========================'.format(acc, metrics))

def dev_model(da_model, dev_loader):
    da_model.eval()
    class_targets, class_preds = [], []
    for step, ipt in enumerate(dev_loader):
        ipt = {k: v.cuda(non_blocking=True) for k, v in ipt.items()}
        class_logits, fea = da_model(ipt)
        class_labels = ipt['labels'].cpu().detach().numpy()
        class_pred = torch.max(class_logits, dim=-1)[1].cpu().detach().numpy()
        class_targets.append(class_labels)
        class_preds.append(class_pred)
    class_targets = np.concatenate(class_targets)
    class_preds = np.concatenate(class_preds)

    class_acc = accuracy_score(class_targets, class_preds)
    return class_acc

def label(args, target='sst5'):
    ''' 根据已有的域样本，再加上目标域样本，利用labeler进行标记
    :param args:
    :param confidence:
    :param target:
    :return:
    '''
    # 加载生成的数据
    with open('datasets/{}/api_generate_{}.txt'.format(args.task, target), 'r') as rf:
        texts = [each.strip() for each in rf.readlines() if each.strip() != '']
    target_frame = pd.read_csv('datasets/{}/{}/test.tsv'.format(args.task, target), sep='	').dropna()
    if args.task in ['NLI']:
        target_texts = 'Premise: ' + target_frame['Premise'].astype(str) + ' Hypothesis: ' + target_frame['Hypothesis'].astype(str)
        target_texts = list(target_texts.to_numpy())[:5000]
    else:
        target_texts = list(target_frame['Text'].values)[:5000]
    tokenizer = AutoTokenizer.from_pretrained(args.LLM_root + 'roberta')
    if target == 'anli':
        texts = target_texts
    else:
        texts = texts + target_texts
    dataset = TextDataset(tokenizer, texts, list(range(len(texts))))  # list(range(len(texts)))为样本索引
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)  # 注意不要shuffle
    # 加载标签标记器
    labeler = torch.load('retriever/{}_labeler.pth'.format(args.source_datasets[args.task])).cuda()

    # 检索，选出域相关样本
    labeler.eval()
    confidences, preds, idxs, csv_datas = [], [], [], []
    for step, ipt in enumerate(dataloader):
        ipt = {k: v.cuda(non_blocking=True) for k, v in ipt.items()}
        logits, fea = labeler(ipt)
        logits = torch.softmax(logits, dim=-1)
        confidence, pred = torch.max(logits, dim=-1)
        confidences.append(confidence.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
        idxs.append(ipt['labels'].detach().cpu().numpy())  # 记录索引用于从检索结果中选取原样本，嘿嘿嘿
        if step % 50 == 0:
            print('=============={}-{}============='.format(step, len(dataloader)))

    preds = np.concatenate(preds)
    confidences = np.concatenate(confidences)
    idxs = np.concatenate(idxs)

    # 根据置信度进行数据筛选
    for i in range(len(preds)):
        csv_datas.append([texts[idxs[i]], preds[i], confidences[i]])
    # 保存高置信度的样本
    print('==================一共标记{}条高置信度样本=================='.format(len(csv_datas)))
    df = pd.DataFrame(csv_datas, columns=["Text", "Label", "Confidence"])
    df.to_csv("retrieve_results/{}.csv".format(target), index=False)

def gpt_label(args, target):
    ''' 使用GPT2进行样本标记
    :return:
    '''
    args.shots = 2
    tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen2-1.5b', trust_remote_code=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/models/Qwen2-1.5b', trust_remote_code=True).half().cuda()

    # 加载数据集【源域数据和目标域数据】
    df_source = pd.read_csv('datasets/{}/{}/train.tsv'.format(args.task, source), sep='	').dropna()
    texts_source = df_source['Text'].tolist()
    labels_source = df_source['Label'].tolist()
    datas = [[texts_source[i], args.label_space[args.task][labels_source[i]]] for i in range(len(labels_source))]

    target_frame = pd.read_csv('datasets/{}/{}/test.tsv'.format(args.task, target), sep='	').dropna()
    target_texts = list(target_frame['Text'].values)
    # generate texts from ChatGPT
    # 加载生成的数据
    with open('datasets/{}/api_generate_{}.txt'.format(args.task, target), 'r') as rf:
        generate_texts = [each.strip() for each in rf.readlines() if each.strip() != '']
    target_texts = target_texts + generate_texts
    target_datas = [[target_texts[i], -100] for i in range(len(target_texts))]
    # 使用源域数据对目标域和生成域进行标记
    # 1. 首先还是先生成random样本
    context_samples = scorer.random_scorer(args, texts_source, labels_source, target_texts, topK=10)
    # 2. 随后生成batch
    ICL_datas, ICL_labels = [], []
    assert len(context_samples) == len(target_texts)
    for test_sample, context in zip(target_datas, context_samples):
        context_shots = []
        for i in range(args.shots):
            for label in context.keys():
                try:
                    text = datas[context[label][i]]
                    text[0] = ' '.join(text[0].split()[:args.max_sen_len])
                    context_shots.append(text)
                except:
                    continue
        # 根据context_shots、test_sample以及对应的instructions构建ICL的prompt
        prompt = ICL_baselines.generate_prompt(args, context_shots, test_sample)
        ICL_datas.append(prompt)
    batches = []
    for i in range(0, len(ICL_datas), args.batch):
        batch = ICL_datas[i:i + args.batch]
        batches.append(batch)
    print('process datasets ({}) with batch ({}) ......'.format(target, len(batches)))
    # 3. 利用gpt2进行少样本预测并获取置信度最高的预测结果作为伪标签数据
    preds, selected_idx, pred_labels, confidences = [], [], [], []
    for step in tqdm(list(range(len(batches)))):
        batch = batches[step]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.cuda()
        with torch.no_grad():
            results = model.generate(input_ids, attention_mask=inputs.attention_mask.cuda(),
                                     pad_token_id=tokenizer.pad_token_id, max_new_tokens=1, output_scores=True,
                                     return_dict_in_generate=True)
        generated_tokens = results.sequences[:, -1]
        scores = results.scores[0]  # 只生成一个新 token，取第一个 logits
        # 计算生成 token 的概率分布
        probs = torch.softmax(scores, dim=-1)  # [16, 50257]
        # 获取生成 token 的索引
        generated_token_indices = generated_tokens
        # 从概率分布中提取生成 token 的置信度
        confidence_scores = probs[torch.arange(probs.size(0)), generated_token_indices]
        # uncertainty_scores = torch.var(scores, dim=-1)
        # 打印生成 token 及其置信度
        for i, (token, confidence) in enumerate(zip(generated_tokens, confidence_scores)):
            token_str = tokenizer.decode(token).strip().lower()
            if token_str == 'entail':
                token_str = 'entailment'
            confidences.append(confidence.item())
            if args.task == 'TD':
                if token_str in args.label_space[args.task][0]:
                    preds.append(0)
                elif token_str in args.label_space[args.task][1]:
                    preds.append(1)
                else:
                    preds.append(100)
            else:
                if token_str in args.label_space[args.task][0]:
                    preds.append(0)
                elif token_str in args.label_space[args.task][1]:
                    preds.append(1)
                elif token_str in args.label_space[args.task][2]:
                    preds.append(2)
                else:
                    preds.append(100)

    # 打印当前iter模型输出结果以及对应标签
    df = pd.DataFrame({
        'Text': target_texts[:len(preds)],
        'Label': preds,
        'Confidence': confidences
    })
    df.to_csv('retrieve_results/gpt_{}.csv'.format(target), index=False)

if __name__ == '__main__':
    args = cfg.Config()
    args.task = 'NLI'
    args.batch = 128
    source = 'mnli'
    target = 'anli'
    # train_labeler(args, source=source)
    label(args, target)