import argparse
import json
import os
from pprint import pprint
from collections import Counter
import torch

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from .mnli import NLIRelationClassifierWithMappingHead, REInputFeatures
from .tacred import *
from .utils import find_optimal_threshold, apply_threshold, f1_score_my, get_tpl_rel_scores

CLASSIFIERS = {"mnli-mapping": NLIRelationClassifierWithMappingHead}


def top_k_accuracy(output, labels, k=5):
    preds = np.argsort(output)[:, ::-1][:, :k]
    return sum(l in p and l > 0 for l, p in zip(labels, preds)) / (labels > 0).sum()


parser = argparse.ArgumentParser(
    prog="run_evaluation",
    description="Run a evaluation for each configuration.",
)
parser.add_argument(
    "input_file",
    type=str,
    default="data/tacred/dev.json",
    help="Dataset file.",
)
parser.add_argument(
    "--config",
    type=str,
    dest="config",
    help="Configuration file for the experiment.",
)
parser.add_argument("--basic", action="store_true", default=False)

args = parser.parse_args()

labels2id = (
    {label: i for i, label in enumerate(TACRED_LABELS)}
    if not args.basic
    else {label: i for i, label in enumerate(TACRED_BASIC_LABELS)}
)
id2Rel = (
    {i: label for i, label in enumerate(TACRED_LABELS)}
    if not args.basic
    else {i: label for i, label in enumerate(TACRED_BASIC_LABELS)}
)

with open(args.input_file, "rt") as f:
    features, labels, is_key = [], [], []
    annotation = []
    lines = json.load(f)
    # lines = lines[:20]
    for line in lines:
        annotation.append(line['relation'])
        line["relation"] = (
            line["relation"] if not args.basic else TACRED_BASIC_LABELS_MAPPING.get(line["relation"], line["relation"])
        )
        features.append(
            REInputFeatures(
                subj="",
                obj="",
                context=line["token"]
                .replace("-LRB-", "(")
                .replace("-RRB-", ")")
                .replace("-LSB-", "[")
                .replace("-RSB-", "]"),
                label=line["relation"],
            )
        )
        
        labels.append(labels2id[line["relation"]])
        is_key.append(line["is_key"])

labels = np.array(labels)
is_key = np.array(is_key)

with open(args.config, "rt") as f:
    config = json.load(f)

LABEL_LIST = TACRED_BASIC_LABELS if args.basic else TACRED_LABELS

for configuration in config:
    n_labels = len(LABEL_LIST)
    os.makedirs(f"experiments/{configuration['name']}", exist_ok=True)
    _ = configuration.pop("negative_threshold", None)
    classifier = CLASSIFIERS[configuration["classification_model"]](negative_threshold=0.0, **configuration)
    output, tpl_score, tpl_list = classifier(
        features,
        batch_size=configuration["batch_size"],
        multiclass=configuration["multiclass"],
    )
    # 在没有处理output之前，可以得到每个关系上的概率
    detail_prob = []
    for i in range(len(output)):
        now = {}
        for j in range(len(output[i])):
            if output[i][j] > 0.0:
                now[id2Rel[j]] = output[i][j]
        # print(now)
        detail_prob.append(now)
        
    #
    if not "use_threshold" in configuration or configuration["use_threshold"]:
        # optimal_threshold, _ = find_optimal_threshold(labels, output)
        print("find optimal threshold")
        optimal_threshold, _ = find_optimal_threshold(is_key, output)
        output_ = apply_threshold(output, threshold=optimal_threshold)
    else:
        os.makedirs(f"boosting/{configuration['name'].replace('/', '_')}", exist_ok=True)
        suffix = args.input_file.split('/')[-1].split('_')[0]
        np.save(f"boosting/{configuration['name'].replace('/', '_')}/output_{suffix}.npy", output)

        print("not find optimal threshold")
        output_ = output.argmax(-1)

    cnt_crt = {}
    cnt_wng = {}
    for i, it in enumerate(output_):
        if id2Rel[it] not in cnt_crt.keys():
            cnt_crt[id2Rel[it]] = []
        if id2Rel[it] not in cnt_wng.keys():
            cnt_wng[id2Rel[it]] = []
        if labels[i] == it:
            cnt_crt[id2Rel[it]].append(i)
        else:
            if id2Rel[labels[i]] not in cnt_wng.keys():
                cnt_wng[id2Rel[labels[i]]] = []
            cnt_wng[id2Rel[labels[i]]].append((i, id2Rel[it]))
            
    if configuration["name"] == "tolog":
        log_dir = os.path.join(configuration["log_dir"], configuration["log_name"])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        crt_log_path = os.path.join(log_dir, "crt.json")
        wng_log_path = os.path.join(log_dir, "wng.json")
        prb_log_path = os.path.join(log_dir, "detail_prob.json")
        with open(crt_log_path, 'a') as f:
            json.dump(cnt_crt, f, indent=4)
        with open(wng_log_path, 'a') as f:
            json.dump(cnt_wng, f, indent=4)
        with open(prb_log_path, 'a') as f:
            json.dump(detail_prob, f, indent=4)
        

    labels_str = []
    for lb in labels:
        labels_str.append(id2Rel[lb])
    output_str = []
    for it in output_:
        output_str.append(id2Rel[it])
    f1, pre, rec, f1_by_relation, prec_by_relation, recall_by_relation = f1_score_my(output_str, labels_str, TACRED_LABELS)
    print("f1, pre, rec: ")
    print(f1, pre, rec)

    tpl_rel_scores = get_tpl_rel_scores(tpl_score, labels, tpl_list)
    slash_pos = -1
    for ch in range(len(args.input_file)-1, 0, -1):
        if args.input_file[ch] == '/':
            slash_pos = ch
            break
    assert(slash_pos != -1)
    data_name = args.input_file[slash_pos+1:-5]
    tmp_path = os.path.join(os.getcwd(), "tpl_scores_"+data_name+".json")
    with open(tmp_path, 'a') as f:
        json.dump(tpl_rel_scores, f, indent=4)

    pre, rec, f1, _ = precision_recall_fscore_support(
        labels, output_, average="micro", labels=list(range(1, n_labels))
    )

    os.makedirs(f"experiments/{configuration['name'].replace('/', '_')}", exist_ok=True)
    suffix = args.input_file.split('/')[-1].split('_')[0]
    np.save(f"experiments/{configuration['name'].replace('/', '_')}/output_{suffix}.npy", output_)
    # np.save(f"experiments/{configuration['pretrained_model'].replace('/', '_')}/labels.npy", labels)
    # 根据output计算每个句子是否为key_stc

    assert(output_.shape[0] == len(is_key) and len(is_key) == len(annotation))
    pred_is_key = []
    for i in range(output_.shape[0]):
        if output_[i] == 0:
            pred_is_key.append(0)
        else:
            pred_is_key.append(1)
    accuracy_crt = 0
    ept_0_accuracy_crt = 0
    ept_0_accuracy_tot = 0
    for i in range(len(is_key)):
        if annotation[i] != '0':
            ept_0_accuracy_tot += 1
        if pred_is_key[i] == is_key[i]:
            accuracy_crt += 1
            if annotation[i] != '0':
                ept_0_accuracy_crt += 1

    accuracy_is_key = float(accuracy_crt) / float(len(is_key))
    ept_0_accuracy_is_key = float(ept_0_accuracy_crt) / float(ept_0_accuracy_tot)
    print("accuracy_is_key")
    print(accuracy_is_key)
    print("ept_0_accuracy_is_key")
    print(ept_0_accuracy_is_key)

    if not "use_threshold" in configuration or configuration["use_threshold"]:
        configuration["all_class_is_key_stc_accuracy"] = accuracy_is_key
        configuration["except_0_class_is_key_stc_accuracy"] = ept_0_accuracy_is_key
        configuration["negative_threshold"] = optimal_threshold
    
    pprint(configuration) # 重新写回
    del classifier
    torch.cuda.empty_cache()


with open(args.config, "wt") as f:
    json.dump(config, f, indent=4)
