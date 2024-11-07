
class Config:
    task = 'TD'
    assert task in ['NLI', 'SA', 'TD']
    LLM = 'llama3.2-3b'   # Qwen2-1.5b Qwen2-7b gpt-j-6b gpt-xl llama3.2-3b
    LLM_root = '/root/autodl-tmp/models/'
    LLM_path = LLM_root + LLM
    shots = 3
    batch = 10
    max_sen_len = 80
    nclass = 3
    score_func = 'simcse'    # 用来执行样本检索的函数: simcse, multi(表示使用相似性和多样性的综合指标)
    use_generate = True     # 在搜索的时候使用生成的样本
    test_datasets = {
        'SA': ['dynasent', 'semeval', 'sst5', 'amazon'],
        'TD': ['adv_civil', 'implicit_hate', 'toxigen', 'civil_comments'],
        'NLI': ['anli', 'contract_nli', 'wanli', 'mnli'],
    }
    source_datasets = {
        'SA': 'amazon',
        'TD': 'civil_comments',
        'NLI': 'mnli'
    }
    label_space = {
        'SA': {0: 'negative', 1: 'positive', 2: 'neutral'},
        'TD': {0: 'benign', 1: 'toxic'},
        'NLI': {0: 'entailment', 1:'neutral', 2:'contradiction'}
    }