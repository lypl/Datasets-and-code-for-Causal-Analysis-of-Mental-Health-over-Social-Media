""" transform csv to json, random split to train and dev"""

import os
import json
import csv
import numpy as np
import pandas as pd
import random
from datetime import datetime
from collections import Counter


file_name = ["IntentSDCNL_Testing", "IntentSDCNL_Training"]

in_dir = "/home/lypl/trldc-main/data/CAMS_raw_data"
# out_dir = "/home/lypl/trldc-main/data/CAMS_json_data"
# out_dir = "/home/lypl/trldc-main/data/CAMS_json_data_0.93"
out_dir = "/home/lypl/trldc-main/data/CAMS_json_data_82"
# out_dir = "/home/lypl/trldc-main/data/CAMS_json_data_91"
# out_dir = "/home/lypl/trldc-main/data/CAMS_json_data_with_addedset"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

out_train_path = os.path.join(out_dir, "IntentSDCNL_Training"+".json")
out_dev_path = os.path.join(out_dir, "IntentSDCNL_Deving"+".json")

random.seed(datetime.now().timestamp())

for fn in file_name:
    in_path = os.path.join(in_dir, fn+".csv")
    read_file = pd.read_csv(in_path, encoding = "ISO-8859-1")
    data = pd.DataFrame(read_file,columns= ['selftext','ANNOTATIONS', 'Interpretations'])
    data = data.rename(columns={'selftext': 'text','ANNOTATIONS':'labels', 'Interpretations':'Interpretations'})
    data= data.convert_dtypes()
    # print(data)
    # input("======")
    data = data.to_dict()
    print(fn)
    cnt = Counter()
    for it in data["text"].keys():
        if type(data["Interpretations"][it]) is not pd._libs.missing.NAType  and len(data["Interpretations"][it]) > 0:
            cnt[str(data["labels"][it])] += 1
    print(cnt)
    # print(type(data))
    # print(data)
    # input("--------")
    # if fn == "IntentSDCNL_Testing":
    #     out_path = os.path.join(out_dir, fn+".json")
    #     with open(out_path, 'w', encoding="ISO-8859-1") as f:
    #         for it in data["text"].keys():
    #             new_row = {
    #                 "text": data["text"][it],
    #                 "labels": [str(data["labels"][it])]
    #             }
    #             # print(it, new_row)
    #             f.write(json.dumps(new_row))
    #             f.write("\n")
    # else:
    #     all_data = {}
    #     train = []
    #     dev = []
    #     for it in data["text"].keys():
    #         new_row = {
    #             "text": data["text"][it],
    #             "labels": [str(data["labels"][it])]
    #         }
    #         if new_row['labels'][0] not in all_data.keys():
    #             all_data[new_row['labels'][0]] = []
    #         all_data[new_row['labels'][0]].append(new_row)
    #     # 拆分train -> dev + train
    #     prop = 0.8
    #     for key in all_data.keys():
    #         random.shuffle(all_data[key])
    #         train_num = int(prop*len(all_data[key]))
    #         train.extend(all_data[key][:train_num])
    #         dev.extend(all_data[key][train_num:])
    #     random.shuffle(train)
    #     random.shuffle(dev)
    #     with open(out_train_path, 'w', encoding="ISO-8859-1") as f:
    #         for dt in train:
    #             f.write(json.dumps(dt))
    #             f.write("\n")
    #     with open(out_dev_path, 'w', encoding="ISO-8859-1") as f:
    #         for dt in dev:
    #             f.write(json.dumps(dt))
    #             f.write("\n")

        # 将 add数据集加入train.json,构建    
        # in_path = os.path.join(in_dir, "added_CAMS_data.csv")
        # read_file = pd.read_csv(in_path, encoding = "ISO-8859-1")
        # data = pd.DataFrame(read_file,columns= ['selftext','cause'])
        # data = data.rename(columns={'selftext': 'text','cause':'labels'})
        # data= data.convert_dtypes()
        # data = data.to_dict()

        # all_data = {}
        # train = []
        # dev = []
        # for it in data["text"].keys():
        #     new_row = {
        #         "text": data["text"][it],
        #         "labels": [str(data["labels"][it])]
        #     }
        #     if new_row['labels'][0] not in all_data.keys():
        #         all_data[new_row['labels'][0]] = []
        #     all_data[new_row['labels'][0]].append(new_row)
        # for key in all_data.keys():
        #     train.extend(all_data[key][:])
        # with open(out_train_path, 'a', encoding="ISO-8859-1") as f:
        #     for dt in train:
        #         if type(dt['text']) == pd._libs.missing.NAType:
        #             continue
        #         f.write(json.dumps(dt))
        #         f.write("\n")
  

