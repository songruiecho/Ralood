# 基于检索增强结果的上下文学习
# 首先，我们要加载已有的检索器，并对总体的外部知识进行检索以及标记，将高置信度【这里的高置信度是指域标签和类别标签都具有高置信度】的样本作为候选样本集合。
# 随后，利用kNN之类的检索方法从候选集中选出与目标与样本test相似的样本作为ICL的演示，并最终进行ICL预测。

import os
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
from api import chat

# 转换numpy.int64为标准的int
def convert_to_builtin_types(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    raise TypeError(f'Not serializable: {repr(obj)}')

def topk_retrieve(args, target='sst5', confidence=0.5):
    '''
    在粗检索得到的候选集中进行topK精排，并构建K-shot样本
    '''
    frame = pd.read_csv("retrieve_results/{}.csv".format(target)).dropna()
    # frame = pd.read_csv("retrieve_results/gpt_{}.csv".format(target)).dropna()
    frame = frame[(frame['Confidence'] > confidence) & (frame['Text'].apply(len) > 10)]

    target_frame = pd.read_csv("datasets/{}/{}/test.tsv".format(args.task, target), sep='	').dropna()
    print(frame.shape, target_frame.shape)
    texts = list(frame['Text'].values)
    labels = list(frame['Label'].values)

    if args.task in ['SA', 'TD']:
        target_texts = list(target_frame['Text'].values)[:5000]
        target_labels = list(target_frame['Label'].values)[:5000]
    elif args.task in ['NLI']:
        target_texts = 'Premise: ' + target_frame['Premise'].astype(str) + ' Hypothesis: ' + target_frame[
            'Hypothesis'].astype(str)
        target_texts = list(target_texts.to_numpy())[:5000]

    if args.score_func == 'simcse':
        all_sim_records = simcse_scorer(args, texts, labels, target_texts)
    if args.score_func == 'multi':
        all_sim_records = sim_div_scorer(args, texts, labels, target_texts)

    with open('topk/{}_{}_{}.json'.format(args.task, args.score_func, target), 'w',
              encoding='utf-8') as f:
        json.dump(all_sim_records, f, default=convert_to_builtin_types, ensure_ascii=False,
                  indent=4)  # indent参数用于增加可读性

# ========================functions for ICL bellow============================= #
# 辅助函数，检查字符是否可打印
def printable(char):
    return char in string.printable

def clean_text(text):
    # 去除非打印字符
    text = ''.join(c for c in text if printable(c))
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace("\\n", " ")
    # 去除或替换其他特殊字符（例如，将所有数字替换为#）
    # text = re.sub(r'\d+', '#', text)

    # 去除或替换Unicode字符（这里只是示例，通常不建议这样做）
    # 注意：这可能会删除所有非ASCII字符，通常不是最佳选择
    # text = ''.join(c for c in text if ord(c) < 128)
    # 去除前导和尾随空格
    text = text.strip()
    return text

def load_LLMs(cfg):
    if 'gpt2' in cfg.LLM or 'opt' in cfg.LLM:
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
    print("(*^_^*) model load finished!!!! ")
    model.eval()
    return model, tokenizer


def generate_prompt(cfg, context_shots, test_sample):
    '''
    :param testset:    test dataset
    :return:
    '''
    Instructions = {
        'SA': 'Solve the sentiment analysis task. Options for sentiment: negative, positive, neutral. \n',
        'TD': 'Solve the toxic detection task. Options for toxicity: benign, toxic. \n',
        'NLI': 'Solve the NLI task. Options for entailment relationship: entailment, neutral, contradiction. \n'
    }
    prompt = Instructions[cfg.task]
    for shot in context_shots:
        if cfg.task == 'NLI':
            prompt = prompt + '{} Prediction: {}\n'.format(' '.join(shot[0].replace('\n', '').split()[:cfg.max_sen_len]), shot[1])
        else:
            prompt = prompt + 'Text: {} Prediction: {}\n'.format(' '.join(shot[0].replace('\n', '').split()[:cfg.max_sen_len]), shot[1])
    test_text = ' '.join(test_sample[0].split(' ')[:cfg.max_sen_len])
    if cfg.task == 'NLI':
        prompt = prompt + '{} Prediction: '.format(test_text)
    else:
        prompt = prompt + 'Text: {} Prediction: '.format(test_text)
    prompt = clean_text(prompt)
    return prompt

def get_batch(cfg, testset, confidence):
    '''
    :param testset:    test dataset
    :return:
    '''
    # 首先加载数据
    ICL_datas, ICL_labels = [], []

    frame = pd.read_csv("retrieve_results/{}.csv".format(testset)).dropna()
    # target = testset
    # frame = pd.read_csv("retrieve_results/gpt_{}.csv".format(target)).dropna()
    frame = frame[(frame['Confidence'] > confidence) & (frame['Text'].apply(len) > 10)]
    target_frame = pd.read_csv("datasets/{}/{}/test.tsv".format(args.task, testset), sep='	').dropna()
    texts = list(frame['Text'].values)
    labels = list(frame['Label'].values)

    if args.task in ['SA', 'TD']:
        target_texts = list(target_frame['Text'].values)[:5000]
        target_labels = list(target_frame['Label'].values)[:5000]
    elif args.task in ['NLI']:
        target_texts = 'Premise: ' + target_frame['Premise'].astype(str) + ' Hypothesis: ' + target_frame[
            'Hypothesis'].astype(str)
        target_texts = list(target_texts.to_numpy())[:5000]
        target_labels = list(target_frame['Label'].values)[:5000]

    print(frame.shape, target_frame.shape)

    datas = [[texts[i], cfg.label_space[cfg.task][labels[i]]] for i in range(len(texts))]
    target_datas = [[target_texts[i], cfg.label_space[cfg.task][target_labels[i]]] for i in range(len(target_texts))]

    file = open('topk/{}_{}_{}.json'.format(cfg.task, cfg.score_func, testset), 'r', encoding='utf-8')
    context_samples = json.load(file)[:5000]
    assert len(context_samples) == len(target_texts)

    for test_sample, context in zip(target_datas, context_samples):
        context_shots = []
        for i in range(cfg.shots):
            for label in context.keys():
                try:
                    text = datas[context[label][i]]
                    text[0] = ' '.join(text[0].split()[:cfg.max_sen_len])
                    context_shots.append(text)
                except:
                    continue
        # 根据context_shots、test_sample以及对应的instructions构建ICL的prompt
        prompt = generate_prompt(cfg, context_shots, test_sample)
        # print(prompt)
        ICL_datas.append(prompt + '\n')
        ICL_labels.append(test_sample[1])
    with open('case/implicit_div.txt', 'w') as wf:
        wf.write('\n'.join(ICL_datas))
    exit(111)
    # 分batch
    batches = []
    for i in range(0, len(ICL_labels), cfg.batch):
        batch = ICL_datas[i:i + cfg.batch]
        batch_labels = ICL_labels[i:i + cfg.batch]  # 标签
        batches.append([batch, batch_labels])
    print('process datasets ({}) with batch ({}) ......'.format(len(ICL_labels), len(batches)))
    return batches

def ICLer(cfg, target, confidence):
    ''' 最终执行上下文学习并得到模型的测试结果
    :return:
    '''
    model, tokenizer = load_LLMs(cfg)
    batches = get_batch(cfg, target, confidence)
    preds, labels = [], []
    for batch in tqdm(batches):
        inputs = tokenizer(batch[0], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        with torch.no_grad():
            generate_ids = model.generate(input_ids.cuda(), attention_mask=inputs.attention_mask.cuda(),
                                          pad_token_id=0, max_new_tokens=5)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for out, inp in zip(output, batch[0]):
            out = out.replace(inp, '').lower()
            if cfg.task == 'TD':
                if cfg.label_space[cfg.task][0] in out:
                    preds.append(0)
                elif cfg.label_space[cfg.task][1] in out:
                    preds.append(1)
                else:
                    preds.append(100)
            else:
                if cfg.label_space[cfg.task][0] in out:
                    preds.append(0)
                elif cfg.label_space[cfg.task][1] in out:
                    preds.append(1)
                elif cfg.label_space[cfg.task][2] in out:
                    preds.append(2)
                else:
                    preds.append(100)
        labels.extend(batch[1])

    swapped_label_space = {value: key for key, value in cfg.label_space[cfg.task].items()}
    labels = [swapped_label_space[each] for each in labels]
    acc = accuracy_score(preds, labels)
    print('===================final acc:{}============='.format(acc))

def ICL_gpt(args, target='sst5', confidence=0.94, test_model='llama'):
    ''' 使用gpt3标记数据的程序，为了防止程序中断，使用追加的方式将数据存储进gpt_results文件中
    :param args:
    :param target:
    :param confidence:
    :return:
    '''
    args.batch = 1
    # topk_retrieve(args, target=target, confidence=confidence)
    target_frame = pd.read_csv("datasets/{}/{}/test.tsv".format(args.task, target), sep='	').dropna()
    if args.task in ['SA', 'TD']:
        target_texts = list(target_frame['Text'].values)[:5000]
        target_labels = list(target_frame['Label'].values)[:5000]
    elif args.task in ['NLI']:
        target_texts = 'Premise: ' + target_frame['Premise'].astype(str) + ' Hypothesis: ' + target_frame[
            'Hypothesis'].astype(str)
        target_texts = list(target_texts.to_numpy())[:5000]
        target_labels = list(target_frame['Label'].values)[:5000]
    if '{}.txt'.format(target) not in os.listdir("{}_results".format(test_model)):
        count = 0
    else:
        with open('{}_results/{}.txt'.format(test_model, target), 'r') as rf:
            already_results = [each.split('\t') for each in rf.readlines()]
        count = len(already_results)     # count=500时结束

    batches = get_batch(args, target, confidence)

    with open('{}_results/{}.txt'.format(test_model, target), 'a') as rf:
        for batch, test_text in tqdm(zip(batches[count:], target_texts[count:])):
            if count >= 500:
                break
            data, label = batch[0][0], batch[1][0]
            pred = chat(data)
            rf.write('\t'.join([test_text.replace('\n', ' '), label, pred])+'\n')    # 按照样本、标签、预测结果的顺序进行封装
            count += 1
        # for data in batch[0]:
        #     print(data, target)
        #     pred = chat(data)
        #     print(pred)
        #     exit(111)


def get_gpt_acc(test_model):
    # 计算每个gpt生成的实验结果中的acc进行比对
    files = os.listdir('{}_results'.format(test_model))
    for file in files:
        with open('{}_results/{}'.format(test_model, file), 'r') as rf:
            datas = [each.strip().split('\t') for each in rf.readlines()][:500]

        count = len(datas)
        right = 0
        for data in datas:
            try:
                if data[1] in data[2]:
                    right += 1
            except:
                continue
        print(file, right/count)



if __name__ == '__main__':
    args = cfg.Config()
    args.task = 'TD'
    # for n in range(5):
    #     args.shots = n+1
    #     for c in [0.96]:
    #         for target in ['sst5', 'dynasent']:
    #             # topk_retrieve(args, target=target, confidence=c)
    #             for model in ['llama3.1-8b']:
    #             # for model in ['opt-6.7b']:
    #                 args.LLM = model
    #                 args.LLM_path = args.LLM_root + args.LLM
    #                 ICLer(args, target=target, confidence=c)
    # topk_retrieve(args, target='implicit_hate', confidence=0.97)
    args.score_func = 'multi'
    ICL_gpt(args, "implicit_hate", 0.97)
    # ICL_gpt(args, "implicit_hate", 0.97)
    # ICL_gpt(args, "toxigen", 0.98)
    # args.task = 'NLI'
    # ICL_gpt(args, "anli", 0.95)
    # ICL_gpt(args, "contract_nli", 0.7)
    # ICL_gpt(args, "wanli", 0.9)
    # get_gpt_acc('llama')