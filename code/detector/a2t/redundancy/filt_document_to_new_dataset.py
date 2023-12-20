''' 带句子label的数据集转换为长文本分类需要的格式的数据集 '''
''' train、dev 处理成根据interpretation构造的新数据集(补类0的结果)，test用打分器得到的新数据 '''
import os, json
import numpy as np
import random
from collections import Counter

def concat_to_text(stcs, annotation):
    ret = {
        'text': ' '.join(stcs),
        'labels': [annotation]
    }
    return ret

"""
''' test '''
in_path = "/media/data/3/lyp/CAM_stc_score_dataset/test_nli.json"
with open(in_path, 'r', encoding='ISO-8859-1') as f:
    raw_test = json.load(f)

# ⭐这里要修改0.7
ori_test_path = "/home/lypl/trldc-main/data/CAMS_json_data_0.7/ori/IntentSDCNL_Testing.json"
ori_text = []
with open(ori_test_path, 'r', encoding='ISO-8859-1') as f:
    for line in f.readlines():
        line = json.loads(line)
        ori_text.append(line)

# ⭐这里要修改
pred_path = "/home/lypl/social_stc_score/a2t/relation_classification/experiments/_media_data_3_lyp_stc_score_ckpt_2_checkpoint-2000/output_test.npy" 
pred = np.load(pred_path)
assert(pred.shape[0] == len(raw_test)) # 还要保证读入test时是按顺序的

test_data = []
tmp_stcs_list = {}
tmp_annotation_list = {}
for i, dt in enumerate(raw_test):
    ori_idx = dt['text_id']
    tmp_annotation_list[ori_idx] = dt['relation']
    if ori_idx not in tmp_stcs_list.keys():
        tmp_stcs_list[ori_idx] = []
    if pred[i] != 0:
        tmp_stcs_list[ori_idx].append(dt['token'])

filted_cnt = Counter()
unfilted_cnt = Counter()
assert(len(ori_text) == len(tmp_stcs_list.keys()))
for idx in tmp_stcs_list.keys():
    if len(tmp_stcs_list[idx]) > 0:
        test_data.append(concat_to_text(tmp_stcs_list[idx], tmp_annotation_list[idx]))
        filted_cnt[tmp_annotation_list[idx]] += 1
    else:
        # 放入原句
        test_data.append(ori_text[idx])
        unfilted_cnt[tmp_annotation_list[idx]] += 1
print("clss, filted_cnt, unfilted_cnt, tot_cnt, filted_ratio")
for clss in filted_cnt.keys():
    print(clss, filted_cnt[clss], unfilted_cnt[clss], filted_cnt[clss]+unfilted_cnt[clss], float(filted_cnt[clss])/float(filted_cnt[clss]+unfilted_cnt[clss]))

# ⭐这里要修改0.7

test_path = os.path.join(test_out_dir, "IntentSDCNL_Testing.json")
print("filted_test")
print(len(test_data))
with open(test_path, 'w', encoding='ISO-8859-1') as f:
    for line in test_data:
        f.write(json.dumps(line))
        f.write("\n")
"""

test_out_dir = "/home/lypl/trldc-main/data/CAMS_json_data_0.7/filted"

''' 把原来train\dev中no_key_stc的样本写回去 '''
train_path = os.path.join(test_out_dir, "IntentSDCNL_Training.json")
dev_path = os.path.join(test_out_dir, "IntentSDCNL_Deving.json")
train_data = []
dev_data = []
with open(train_path, 'r', encoding='ISO-8859-1') as f:
    for line in f.readlines():
        line = json.loads(line)
        train_data.append(line)
with open(dev_path, 'r', encoding='ISO-8859-1') as f:
    for line in f.readlines():
        line = json.loads(line)
        dev_data.append(line)
print("dev_has_key_stc_example")
print(len(dev_data))
print("train_has_key_stc_example")
print(len(train_data))


''' dev '''
in_path = "/media/data/3/lyp/CAM_stc_score_dataset/dev_no_key_stc_nli.json"
with open(in_path, 'r', encoding='ISO-8859-1') as f:
    raw_test = json.load(f)
# ⭐这里要修改-dev的
pred_path = "/home/lypl/social_stc_score/a2t/relation_classification/experiments/all_no_key_stc/50/output_dev.npy" 
pred = np.load(pred_path)
print(pred.shape, pred.shape[0], len(raw_test))
assert(pred.shape[0] == len(raw_test)) # 还要保证读入test时是按顺序的
tmp_stcs_list = {}
tmp_all_stcs_list = {}
tmp_annotation_list = {}
dev_annotated_num = Counter()
dev_not_annotated_num = Counter()
for i, dt in enumerate(raw_test):
    ori_idx = dt['text_id']
    tmp_annotation_list[ori_idx] = dt['relation']
    if ori_idx not in tmp_stcs_list.keys():
        tmp_stcs_list[ori_idx] = []
        tmp_all_stcs_list[ori_idx] = []
    tmp_all_stcs_list[ori_idx].append(dt['token'])
    if pred[i] != 0:
        tmp_stcs_list[ori_idx].append(dt['token'])

dev_ant_key_stc_cnt = 0

for idx in tmp_stcs_list.keys():
    if tmp_annotation_list[idx] != '0':
        if len(tmp_stcs_list[idx]) > 0:
            dev_ant_key_stc_cnt += 1
            dev_data.append(concat_to_text(tmp_stcs_list[idx], tmp_annotation_list[idx])) #这样不会得到空数据
            dev_annotated_num[tmp_annotation_list[idx]] += 1
        if len(tmp_stcs_list[idx]) == 0:
            dev_not_annotated_num[tmp_annotation_list[idx]] += 1
        # dev_data.append(concat_to_text(tmp_stcs_list[idx], tmp_annotation_list[idx])) #这样可得到空数据
print("dev_ant_key_stc_cnt")
print(dev_ant_key_stc_cnt)
print("class, dev_annotated_num, dev_unannotated_num, dev_tot, dev_annotated_ratio")
for clss in dev_annotated_num.keys():
    tot = dev_annotated_num[clss]+dev_not_annotated_num[clss]
    print(clss, dev_annotated_num[clss], dev_not_annotated_num[clss], tot, float(dev_annotated_num[clss])/float(tot))
print("filted_dev")
print(len(dev_data))
with open(dev_path, 'w', encoding='ISO-8859-1') as f:
    for line in dev_data:
        f.write(json.dumps(line))
        f.write("\n")


''' train '''
in_path = "/media/data/3/lyp/CAM_stc_score_dataset/train_no_key_stc_nli.json"
with open(in_path, 'r', encoding='ISO-8859-1') as f:
    raw_test = json.load(f)
# ⭐这里要修改-train的
pred_path = "/home/lypl/social_stc_score/a2t/relation_classification/experiments/all_no_key_stc/50/output_train.npy" 
pred = np.load(pred_path)
assert(pred.shape[0] == len(raw_test)) # 还要保证读入test时是按顺序的
tmp_stcs_list = {}
tmp_all_stcs_list = {}
tmp_annotation_list = {}
train_annotated_num = Counter()
train_not_annotated_num = Counter()
for i, dt in enumerate(raw_test):
    ori_idx = dt['text_id']
    tmp_annotation_list[ori_idx] = dt['relation']
    if ori_idx not in tmp_stcs_list.keys():
        tmp_stcs_list[ori_idx] = []
        tmp_all_stcs_list[ori_idx] = []
    tmp_all_stcs_list[ori_idx].append(dt['token'])
    if pred[i] != 0:
        tmp_stcs_list[ori_idx].append(dt['token'])
    
train_ant_key_stc_cnt = 0
for idx in tmp_stcs_list.keys():
    if tmp_annotation_list[idx] != '0':
        if len(tmp_stcs_list[idx]) > 0:
            train_ant_key_stc_cnt += 1
            train_data.append(concat_to_text(tmp_stcs_list[idx], tmp_annotation_list[idx])) #这样不会得到空数据
            train_annotated_num[tmp_annotation_list[idx]] += 1
        if len(tmp_stcs_list[idx]) == 0:
            train_annotated_num[tmp_annotation_list[idx]] += 1
        # train_data.append(concat_to_text(tmp_stcs_list[idx], tmp_annotation_list[idx])) #这样可得到空数据
print("train_ant_key_stc_cnt")
print(train_ant_key_stc_cnt)
print("class, train_annotated_num, train_unannotated_num, train_tot, train_annotated_ratio")
for clss in train_annotated_num.keys():
    tot = train_annotated_num[clss]+train_not_annotated_num[clss]
    print(clss, train_annotated_num[clss], train_not_annotated_num[clss], tot, float(train_annotated_num[clss])/float(tot))

print("filted_train")
print(len(train_data))
with open(train_path, 'w', encoding='ISO-8859-1') as f:
    for line in train_data:
        f.write(json.dumps(line))
        f.write("\n")

