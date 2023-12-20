""" transform csv to json, random split to train and dev """
''' 涉及的文件目录：/media/data/3/lyp/CAM_stc_score_dataset  '''
import os
import json
import math
import csv
import numpy as np
import pandas as pd
import random
from datetime import datetime

def a_is_subseq_of_b(s, t):
    def match(token_a, token_b):
        if token_a == token_b:
            return True
        
        if len(token_a) > len(token_b):
            token_a, token_b = token_b, token_a
        pt_a = 0
        pt_b = 0
        while(pt_a < len(token_a) and pt_b < len(token_b)):
            if token_a[pt_a] == token_b[pt_b]:
                pt_a += 1
            pt_b +=1
        if pt_a >= math.ceil(0.7*len(token_a)):
            return True
        return False

    a = s.split()
    b = t.split()
    cnt = 0
    pt_a = 0
    pt_b = 0
    while(pt_a < len(a) and pt_b < len(b)):
        if match(a[pt_a], b[pt_b]):
            pt_a += 1
        pt_b +=1
    if pt_a >= int(0.7*len(a)):
        return True
    return False
    


def split_document(test):
    tmp = test.split('\n')
    tmp_ret = []
    for sub_stc in tmp:
        tmp_ret.extend(sub_stc.split('.'))
    ret = []
    for stc in tmp_ret:
        stc = stc.strip()
        if stc == "":
            continue
        ret.append(stc)
    return ret

def split_interpretations(interpretations):
    tmp_ret = interpretations.split(',')
    ret = []
    for it in tmp_ret:
        it = it.strip()
        if it == "":
            continue
        ret.append(it)
    return ret

def judge(stc, itps):
    for itp in itps:
        if a_is_subseq_of_b(itp, stc):
            return True
        else:
            return False
            

def get_item(text, annotations, interpretations):
    stcs = split_document(text)
    length = -1
    wide_flag = False
    for stc in stcs:
        length = max(length, len(stc.split()))
    labels = []
    ipts = interpretations
    if ipts != "":
        ipts = split_interpretations(interpretations)
        for stc in stcs:
            if judge(stc, ipts):
                labels.append(1)
            else:
                labels.append(0)
        assert(len(labels) == len(stcs))
        if sum(labels) > len(ipts):
            wide_flag = True
        # assert(sum(labels) <= len(ipts))

    ret = {
        'text': text,
        'sentences': stcs,
        'labels': labels,
        'interpretations': ipts,
        'annotation': annotations
    }
    return ret, length, wide_flag


file_name = ["IntentSDCNL_Testing", "IntentSDCNL_Training"]
prop = 0.7
in_dir = "/home/lypl/trldc-main/data/CAMS_raw_data"
test_out_dir = "/media/data/3/lyp/CAM_stc_score_dataset"
train_dev_out_dir = "/media/data/3/lyp/CAM_stc_score_dataset/prop_"+str(prop)

os.makedirs(test_out_dir, exist_ok=True)
os.makedirs(train_dev_out_dir, exist_ok=True)

random.seed(datetime.now().timestamp())
all_data = {}
train = []
dev = []
all_test_data = []
for fn in file_name:
    in_path = os.path.join(in_dir, fn+".csv")
    read_file = pd.read_csv(in_path, encoding = "ISO-8859-1")
    data = pd.DataFrame(read_file,columns= ['selftext','ANNOTATIONS', 'Interpretations'])
    data = data.rename(columns={'selftext': 'text','ANNOTATIONS':'labels', 'Interpretations': 'interpretations'})
    data= data.convert_dtypes()
    # print(data)
    # input("======")
    data = data.to_dict()
    # print(type(data))
    # print(data)
    # input("--------")
    
    mx_len = -1
    wide_cnt = 0
    for it in data["text"].keys():
        text = data["text"][it]
        annotations = [str(data["labels"][it])]
        interpretations = data["interpretations"][it] if type(data["interpretations"][it]) == str else ""
        new_row, length, wide_flag = get_item(text, annotations, interpretations)
        mx_len = max(mx_len, length)
        if wide_flag:
            wide_cnt += 1

        if fn == "IntentSDCNL_Testing":
            all_test_data.append(new_row)
        else:
            if new_row['annotation'][0] not in all_data.keys():
                all_data[new_row['annotation'][0]] = []
            all_data[new_row['annotation'][0]].append(new_row)

    print("mx_len")
    print(mx_len)
    print("wide_cnt")
    print(wide_cnt)
    if fn == "IntentSDCNL_Testing":
        out_path = os.path.join(test_out_dir, "test.json")
        with open(out_path, 'w', encoding="ISO-8859-1") as f:
            json.dump(all_test_data, f)
    else:
        out_train_path = os.path.join(train_dev_out_dir, "train.json")
        out_dev_path = os.path.join(train_dev_out_dir, "dev.json")
        # 拆分train -> dev + train
        for key in all_data.keys():
            random.shuffle(all_data[key])
            train_num = int(prop*len(all_data[key]))
            train.extend(all_data[key][:train_num])
            dev.extend(all_data[key][train_num:])
        random.shuffle(train)
        random.shuffle(dev)
        with open(out_train_path, 'w', encoding="ISO-8859-1") as f:
            for dt in train:
                f.write(json.dumps(dt))
                f.write("\n")
        with open(out_dev_path, 'w', encoding="ISO-8859-1") as f:
            for dt in dev:
                f.write(json.dumps(dt))
                f.write("\n")
print("prop")
print(prop)
print("len(all_test_data)")
print(len(all_test_data))
print("len(train)")
print(len(train))
print("len(dev)")
print(len(dev))

# ⭐这里要改
classifier_output_dir = "/home/lypl/trldc-main/data/CAMS_json_data_"+str(prop)
os.makedirs(classifier_output_dir, exist_ok=True)

''' 生成用于训练长文本分类器的原始train、dev、test，和用于训练打分器同样的分割'''
ori_classifier_output_dir = os.path.join(classifier_output_dir, "ori")
os.makedirs(ori_classifier_output_dir, exist_ok=True)
classifier_train_path = os.path.join(ori_classifier_output_dir, "IntentSDCNL_Training.json")
classifier_dev_path = os.path.join(ori_classifier_output_dir, "IntentSDCNL_Deving.json")
classifier_test_path = os.path.join(ori_classifier_output_dir, "IntentSDCNL_Testing.json")
with open(classifier_train_path, 'w', encoding='ISO-8859-1') as f:
    for line in train:
        f.write(json.dumps({
            'text': line["text"],
            'labels': line['annotation']
        }))
        f.write("\n")
with open(classifier_dev_path, 'w', encoding='ISO-8859-1') as f:
    for line in dev:
        f.write(json.dumps({
            'text': line["text"],
            'labels': line['annotation']
        }))
        f.write("\n")
with open(classifier_test_path, 'w', encoding='ISO-8859-1') as f:
    for line in all_test_data:
        f.write(json.dumps({
            'text': line["text"],
            'labels': line['annotation']
        }))
        f.write("\n")


''' 生成用于训练长文本分类器的缩短后的train、dev，和用于训练打分器同样的分割。只写入有key stc的部分(这样很差)，还是全都写入'''
filted_classifier_output_dir = os.path.join(classifier_output_dir, "filted")
os.makedirs(filted_classifier_output_dir, exist_ok=True)
filted_classifier_train_path = os.path.join(filted_classifier_output_dir, "IntentSDCNL_Training.json")
filted_classifier_dev_path = os.path.join(filted_classifier_output_dir, "IntentSDCNL_Deving.json")
filted_classifier_test_path = os.path.join(filted_classifier_output_dir, "IntentSDCNL_Testing.json")

train_no_key_stc = []
dev_no_key_stc = []
train_have_key_stc = []
dev_have_key_stc = []
with open(filted_classifier_dev_path, 'w', encoding='ISO-8859-1') as f:
    for line in dev:
        if '0' in line['annotation']:
            dev_no_key_stc.append(line)
            f.write(json.dumps({
                'text': line["text"],
                'labels': line['annotation']
            }))
            f.write("\n")
        else:
            tmp_stcs = []
            for i in range(len(line['labels'])):
                if line['labels'][i] == 1:
                    tmp_stcs.append(line['sentences'][i])
            if len(tmp_stcs) > 0:
                dev_have_key_stc.append(line)
                f.write(json.dumps({
                    'text': ' '.join(tmp_stcs),
                    'labels': line['annotation']
                }))
                f.write("\n")

                # 将原句也放进来，筛选后的句子做数据增强
                f.write(json.dumps({
                    'text': line["text"],
                    'labels': line['annotation']
                }))
                f.write("\n")
            else:
                dev_no_key_stc.append(line)
                f.write(json.dumps({
                    'text': line["text"],
                    'labels': line['annotation']
                }))
                f.write("\n")

with open(filted_classifier_train_path, 'w', encoding='ISO-8859-1') as f:
    for line in train:
        if '0' in line['annotation']:
            train_no_key_stc.append(line)
            f.write(json.dumps({
                'text': line["text"],
                'labels': line['annotation']
            }))
            f.write("\n")
        else:
            tmp_stcs = []
            for i in range(len(line['labels'])):
                if line['labels'][i] == 1:
                    tmp_stcs.append(line['sentences'][i])
            if len(tmp_stcs) > 0:
                train_have_key_stc.append(line)
                f.write(json.dumps({
                    'text': ' '.join(tmp_stcs),
                    'labels': line['annotation']
                }))
                f.write("\n")

                # 将原句也放进来，筛选后的句子做数据增强
                f.write(json.dumps({
                    'text': line["text"],
                    'labels': line['annotation']
                }))
                f.write("\n")
            else:
                train_no_key_stc.append(line)
                f.write(json.dumps({
                    'text': line["text"],
                    'labels': line['annotation']
                }))
                f.write("\n")

# 将ori_test写入filted_dir
with open(filted_classifier_test_path, 'w', encoding='ISO-8859-1') as f:
    for line in all_test_data:
        f.write(json.dumps({
            'text': line["text"],
            'labels': line['annotation']
        }))
        f.write("\n")

    
''' 把train_0和dev_0写入和test一样的路径 '''
class_0_out_dir = test_out_dir
train_0_output_path = os.path.join(class_0_out_dir, 'train_no_key_stc.json')
train_1_output_path = os.path.join(class_0_out_dir, 'train_have_key_stc.json')
dev_0_output_path = os.path.join(class_0_out_dir, 'dev_no_key_stc.json')
dev_1_output_path = os.path.join(class_0_out_dir, 'dev_have_key_stc.json')
with open(train_0_output_path, 'w', encoding='ISO-8859-1') as f:
    json.dump(train_no_key_stc, f)
with open(dev_0_output_path, 'w', encoding='ISO-8859-1') as f:
    json.dump(dev_no_key_stc, f)
with open(train_1_output_path, 'w', encoding='ISO-8859-1') as f:
    json.dump(train_have_key_stc, f)
with open(dev_1_output_path, 'w', encoding='ISO-8859-1') as f:
    json.dump(dev_have_key_stc, f)