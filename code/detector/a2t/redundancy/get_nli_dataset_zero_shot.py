''' 给每个类别从模板和verbalizer中随机生成一些模板，并构造nli训练数据 '''

import os
import json
import math
import csv
import numpy as np
import pandas as pd
import random
from datetime import datetime

''' 给每个类生成模板（随机） '''
LABEL_WORDS = {
    '0': ["no reason"],
    '1': ['body shame', 'physical abuse', 'sexual abuse', 'emotional abuse'],
    '2': ['loans', 'poverty', 'financial problem', 'unemployment', 'bully', ' gossip', 'bad job', 'poor grades', 'educational problem', 'dictatorial management'],
    '3': ['drug', 'medication', 'tumor', 'cancer', 'disease', 'alcohol addiction', 'smoke'],
    '4': ['bad parenting', 'breakup', 'divorce', 'mistrust', 'jealousy', 'betrayal', 'conflict', 'fight', 'childhood trauma'],
    '5': ['worthless', 'lonely', 'tired  of daily', 'powerless', 'normless', ' isolation', 'estrangement'],
}

random.seed(datetime.now().timestamp())

def get_tpl_from_file(file):
    span_list = [file['template'][1]["content"].strip()]
    span_list.append("{label_words}")
    span_list.append(file['template'][3]["content"].strip())
    return ' '.join(span_list)

def info_to_dict(sentence1, sentence2, label):
    ret = {
        "sentence1": sentence1,
        "sentence2": sentence2,
        "label": label
    }
    return ret

def get_nli_data(raw_data, tpl_mapping):
    '''
    annotation == 0:
        每一个句子 -> 类0的某一个tpl -> 2
        每一个句子 -> 其他所有类的某1个tpl -> 0
        [应该跳过]
    annotation != 0:
        len(labels) > 0:[有interpretations]
            label = 1 的句子 -> 对应类的所有tpl拼接 -> 2(蕴含)
            label = 1 的句子 -> 其他所有类的某2个tpl拼接 -> 0(对立)
            label = 0 的句子 -> 对应类的所有tpl拼接 -> 1(无关)
        len(labels) == 0:
            跳过
    '''
    data = []
    for line in raw_data:
        if len(line["sentences"]) == 0:
            continue
        annotation = line["annotation"]
        if '0' in annotation:
            # no reason example
            annotation = "0"
            # for stc in line["sentences"]:
            #     tpl_idx = random.randint(0, 9)
            #     data.append(info_to_dict(stc, tpl_mapping['0'][tpl_idx], 2))
            #     other_clss_idx = random.randint(1, 5)
            #     data.append(info_to_dict(stc, tpl_mapping[str(other_clss_idx)][tpl_idx], 0))
            continue
            
        else:
            # positive example
            annotation = annotation[0]
            stcs = line["sentences"]
            lbls = line["labels"]
            for i in range(len(lbls)):
                if lbls[i] == 1:
                    for tpl in tpl_mapping[annotation]:
                        data.append(info_to_dict(stcs[i], tpl, 2))
                    for clss in tpl_mapping.keys():
                        if clss == annotation:
                            continue
                        tpl_idx_1 = random.randint(0, len(tpl_mapping[clss])-1)
                        if clss != '0':
                            while(True):
                                tpl_idx_2 = random.randint(0, len(tpl_mapping[clss])-1)
                                if tpl_idx_1 != tpl_idx_2:
                                    break
                        data.append(info_to_dict(stcs[i], tpl_mapping[clss][tpl_idx_1], 0))
                        if clss != '0':
                            data.append(info_to_dict(stcs[i], tpl_mapping[clss][tpl_idx_2], 0))
                else:
                    for tpl in tpl_mapping[annotation]:
                        data.append(info_to_dict(stcs[i], tpl, 1))
    
    return data

templates_list = []
templates_dir = os.path.join("/home/lypl/PromptBoosting/templates/full_templates/t5_sorted_social")

file_list = os.listdir(templates_dir)
for file_name in file_list:
    path = os.path.join(templates_dir, file_name)
    with open(path, 'r') as f:
        tpl_file = json.load(f)
        tpl = get_tpl_from_file(tpl_file)
        templates_list.append(tpl)

all_tpl_mappings = []

''' 模板每个类len(label_words)个,所有类共享从模板pool里随机选一个tpl '''
# template_mapping = {}
# tpl_idx = random.randint(0, len(templates_list)-1)
# for clss in LABEL_WORDS.keys():
#     template_mapping[clss] = []
#     for i in range(len(LABEL_WORDS[clss])):
#         label_word_idx = random.randint(0, len(LABEL_WORDS[clss])-1)
#         now_tpl = templates_list[tpl_idx].format(label_words=LABEL_WORDS[clss][i])
#         template_mapping[clss].append(now_tpl)
# all_tpl_mappings.append(template_mapping)
''' 模板每个类len(label_words)个,各个类分别从模板pool里随机选一个tpl '''
template_mapping = {}
for clss in LABEL_WORDS.keys():
    template_mapping[clss] = []
    tpl_idx = random.randint(0, len(templates_list)-1)
    for i in range(len(LABEL_WORDS[clss])):
        label_word_idx = random.randint(0, len(LABEL_WORDS[clss])-1)
        now_tpl = templates_list[tpl_idx].format(label_words=LABEL_WORDS[clss][i])
        template_mapping[clss].append(now_tpl)
all_tpl_mappings.append(template_mapping)

tpl_map_path = os.path.join(os.getcwd(), 'templates_mappings.json')
with open(tpl_map_path, 'w') as f:
    json.dump(all_tpl_mappings, f, indent=4)

'''  '''
now_config = []
now_config.append({
    "name": "roberta-large-mnli",
    "classification_model": "mnli-mapping",
    "pretrained_model": "/media/data/3/lyp/roberta-large-mnli",
    "batch_size": 2,
    "multiclass": True,
    "use_cuda": True,
    "half": True,
    "use_threshold": False,
    "entailment_position": 2,
    "labels": [
        "0",
        "1",
        "2",
        "3", 
        "4",
        "5"
    ],
    "template_mapping": all_tpl_mappings[0],
})
config_path = os.path.join(os.getcwd(), 'configs', 'zero-shot', "config_roberta.json")
with open(config_path, 'w') as f:
    json.dump(now_config, f, indent=4)





''' 根据各个template_mapping 构造对应的nli训练数据 '''
# in_train_path = os.path.join("/media/data/3/lyp/CAM_stc_score_dataset/prop_0.7", "train.json") # ⭐每次构造这里要改
# in_dev_path = os.path.join("/media/data/3/lyp/CAM_stc_score_dataset/prop_0.7", "dev.json") # ⭐每次构造这里要改
# nli_data_output_dir = "/media/data/3/lyp/CAM_stc_score_dataset/for_NLI_train/prop_0.7" # ⭐每次构造这里要改
# os.makedirs(nli_data_output_dir, exist_ok=True)
# raw_train_data = []
# raw_dev_data = []
# with open(in_train_path, 'r', encoding="ISO-8859-1") as f:
#     for line in f.readlines():
#         line = json.loads(line)
#         raw_train_data.append(line)

# with open(in_dev_path, 'r', encoding="ISO-8859-1") as f:
#     for line in f.readlines():
#         line = json.loads(line)
#         raw_dev_data.append(line)

# for i in range(5):
#     now_template_mapping = all_tpl_mappings[i]
#     train_data = get_nli_data(raw_train_data, template_mapping)
#     dev_data = get_nli_data(raw_dev_data, template_mapping)
#     now_output_dir = os.path.join(nli_data_output_dir, str(i))
#     os.makedirs(now_output_dir, exist_ok=True)
#     train_path = os.path.join(now_output_dir, 'train.json')
#     dev_path = os.path.join(now_output_dir, 'dev.json')
#     with open(train_path, 'w', encoding="ISO-8859-1") as f:
#         for line in train_data:
#             f.write(json.dumps(line))
#             f.write("\n")
#     with open(dev_path, 'w', encoding="ISO-8859-1") as f:
#         for line in dev_data:
#             f.write(json.dumps(line))
#             f.write("\n")
#     print("i")
#     print(i)
#     print("len(train_data)")
#     print(len(train_data))
#     print("len(dev_data)")
#     print(len(dev_data))