import random

import requests
import json
from generater import *

def chat(prompt):
    url = "https://api.xi-ai.cn/v1/chat/completions"
    api_key = [
        'sk-xvu9m3Oxwo9RSt167718D2D1Fb464902BdF138C1F388C749',
        'sk-HMjMaMLCYxpscmYoC3019dFe884149B6977066562bA5Cd4b',
        'sk-9UOnq9q2aseQrOldBb0aFdBb3b7342A89881587c0e54B9A8',
        'sk-4vlbSD6DlZFYjMtXE1A3772f35A440769a2761B72b42395f'
    ]
    random_key = random.choice(api_key)
    payload = json.dumps({
        # "model": "gpt-3.5-turbo",
        "model": "qwen2-72b-instruct",
        "messages": [
            {
               "role": "user",
               "content": prompt
            },

        ],
        "max_tokens": 5,
    })
    headers = {
        'Accept': 'application/json',
        'Authorization': 'Bearer {}'.format(random_key),
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response = json.loads(response.text)
    try:
        text = response['choices'][0]['message']['content']
    except:
        print(response)
        text = ''
    return text


def sample_generate_by_api(args, target, sampled_samles=2000):
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
    if args.task in ['SA', 'TD']:
        texts = list(df_target['Text'].values)
    elif args.task in ['NLI']:
        texts = 'Premise: ' + df_target['Premise'].astype(str) + '\nHypothesis: ' + df_target['Hypothesis'].astype(str)
        texts = list(texts.to_numpy())

    # 创建一个字典来存储每个cluster_label对应的列表
    groups = {label: [] for label in df['cluster_label']}
    # 将数据分配到对应的组中
    for _, row in df.iterrows():
        groups[row['cluster_label']].append(row['text_id'])
    for i in tqdm(range(sampled_samles)):
        data = [random.choice(groups[key]) for key in groups.keys()]
        datas.append(data)
        # 获取样本的prompt

    for data in tqdm(datas):
        prompt = 'Please generate similar texts according to the following samples:\n'
        # prompt += '\n'.join([texts[each].replace('\n', ' ') for each in data[:5]])
        prompt += '\n'.join([texts[each] for each in data[:5]])
        prompt = prompt + '\n'
        result = chat(prompt)
        results = result.split('\n')

        with open('datasets/{}/api_generate_{}.txt'.format(args.task, target), 'a') as wf:
            wf.write('\n'.join(results))

def nli_sample_generate_by_api(args, target, sampled_samles=2000, n_cluster=3):
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
    if args.task in ['SA', 'TD']:
        texts = list(df_target['Text'].values)
    elif args.task in ['NLI']:
        texts = 'Premise: ' + df_target['Premise'].astype(str) + '\nHypothesis: ' + df_target['Hypothesis'].astype(str)
        texts = list(texts.to_numpy())

    # 创建一个字典来存储每个cluster_label对应的列表
    groups = {label: [] for label in df['cluster_label']}
    # 将数据分配到对应的组中
    for _, row in df.iterrows():
        groups[row['cluster_label']].append(row['text_id'])
    for i in tqdm(range(sampled_samles)):
        data = [random.choice(groups[key]) for key in groups.keys()]
        datas.append(data)
        # 获取样本的prompt

    for data in tqdm(datas[133+287+73+98+294+149+200+323+147+197+68:]):
        prompt = 'Please generate similar Premise and Hypothesis pairs according to the following samples:\n'
        # 清除text中的多余空格
        prompt += '\n'.join([texts[each].replace('\n', ' ').replace(' Hypothesis: ', '\nHypothesis: ') for each in data[:n_cluster]])
        prompt = prompt + '\n'
        result = chat(prompt)
        time.sleep(0.2)
        results = []
        Premise, Hypothesis = [], []
        pattern_hyp = r'^Hypothesis:.*'
        pattern_pre = r'^Premise:.*'
        # 使用re.findall查找所有匹配的行
        matches = re.findall(pattern_pre, result, re.MULTILINE)
        for match in matches:
            Premise.append(match)
        matches = re.findall(pattern_hyp, result, re.MULTILINE)
        for match in matches:
            Hypothesis.append(match)
        if len(Premise) == len(Hypothesis):   # 假设是一一对应的
            for p, h in zip(Premise, Hypothesis):
                results.append('{} {}'.format(p, h))
        with open('datasets/{}/api_generate_{}_raw.txt'.format(args.task, target), 'a') as wf:
            wf.write('\n'.join(results))


def process_generate_results(args, target):
    with open('datasets/{}/api_generate_{}_raw.txt'.format(args.task, target), 'r') as rf:
        datas = [each.strip() for each in rf.readlines()]

    new_datas = []
    # 正则表达式模式
    premise_pattern = r'(?<=Premise:\s)(.*?)(?=\s*Hypothesis:)'
    hypothesis_pattern = r'(?<=Hypothesis:\s)(.*?)(?=\s*Premise:)'

    for text in datas:
        # 查找所有匹配的Premise
        premises = re.findall(premise_pattern, text, re.DOTALL)
        # 查找所有匹配的Hypothesis（这里假设每个Premise后都紧跟着一个Hypothesis）
        hypotheses = re.findall(hypothesis_pattern, text)
        # 由于我们假设每个Premise后都紧跟着一个Hypothesis，所以可以直接配对
        for premise, hypothesis in zip(premises, hypotheses):
            new_datas.append('Premise: '+premise.replace("Hypothesis: ", '').strip()+ ' Hypothesis: ' + hypothesis.replace("Premise: ", '').strip())

    # 正则表达式模式
    premise_pattern = r'(?<=Premise:\s)(.*?)(?=\s*Hypothesis:)'
    hypothesis_pattern = r'(?<=Hypothesis:\s)(.*?)(?=\s*Premise:)'

    for text in datas:
        # 查找所有匹配的Premise
        premises = re.findall(premise_pattern, text, re.DOTALL)
        # 查找所有匹配的Hypothesis（这里假设每个Premise后都紧跟着一个Hypothesis）
        hypotheses = re.findall(hypothesis_pattern, text)
        # 由于我们假设每个Premise后都紧跟着一个Hypothesis，所以可以直接配对
        for premise, hypothesis in zip(premises, hypotheses):
            ttt = 'Premise: ' + premise.replace("Hypothesis: ", '').strip() + ' Hypothesis: ' + hypothesis.replace(
                    "Premise: ", '').strip()
            if ttt not in new_datas:
                new_datas.append(ttt)

    for each in new_datas:
        print(each)

    with open('datasets/{}/api_generate_{}.txt'.format(args.task, target), 'w') as wf:
        wf.write('\n'.join(new_datas))



if __name__ == '__main__':
    args = cfg.Config()
    args.task = 'NLI'
    target = 'contract_nli'
    # cluster(args, target, n_cluster=5)
    # nli_sample_generate_by_api(args, target, n_cluster=3)
    # process_generate_results(args, target)
    # process_generate_results(args, target)
