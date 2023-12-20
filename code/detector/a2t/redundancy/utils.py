import numpy as np
import os
import math
import json
from sklearn.metrics import f1_score, precision_recall_fscore_support
from collections import Counter

def get_tpl_rel_scores(tpl_score, labels, tpl_list):
    ret = {}
    down_ret = {}
    cnt = Counter()
    fin_ret = {}
    
    for rel in labels:
        cnt[rel] += 1
    for i, tpl in enumerate(tpl_list):
        ret[tpl] = {}
        fin_ret[tpl] = {}
        down_ret[tpl] = 0.0
        for rel in range(max(labels)+1):
            ret[tpl][rel] = 0.0
        for j in range(tpl_score.shape[0]):
            now_label = labels[j]
            ret[tpl][now_label] += tpl_score[j, i]
        for rel in range(max(labels)+1):
            if cnt[rel] > 0:
                ret[tpl][rel] /= float(cnt[rel])
                down_ret[tpl] += math.exp(ret[tpl][rel])
            else:
                ret[tpl][rel] = -1
        # 若将ret返回，则是config9第一个的结果，好的非常好，差的非常差
        for rel in range(max(labels)+1):
            if ret[tpl][rel] == -1:
                fin_ret[tpl][rel] = -1
            else:
                fin_ret[tpl][rel] = math.log(math.exp(ret[tpl][rel]) / down_ret[tpl])
        return fin_ret
    
    

def f1_score_(labels, preds, n_labels=42):
    return f1_score(labels, preds, labels=list(range(1, n_labels)), average="micro")

def accuracy_(labels, pred):
    pred_is_key = []
    for i in range(pred.shape[0]):
        if pred[i] == 0:
            pred_is_key.append(0)
        else:
            pred_is_key.append(1)
    assert(len(pred_is_key) == labels.shape[0])
    crt = 0
    for i in range(labels.shape[0]):
        if pred_is_key[i] == labels[i]:
            crt += 1
    accuracy_is_key = float(crt) / float(len(labels))
    return accuracy_is_key
    

def f1_score_my(res, labels, relations):
    def judge(string):
        if string in ["NA", "no_relation"]:
            return 0
        return 1
    correct_by_relation = Counter()
    guess_by_relation = Counter()
    gold_by_relation = Counter()

    for i in range(len(labels)):
        guess = res[i]
        gold = labels[i]
        # print("guess, gold")
        # print(guess, gold)
        # print("judge(guess), judge(gold)")
        # print(judge(guess), judge(gold))
        # print("------")
        if judge(gold) == 0 and judge(guess) == 0:
            continue
        if judge(gold) == 0 and judge(guess) != 0:
            guess_by_relation[guess] += 1
        if judge(gold) != 0 and judge(guess) == 0:
            gold_by_relation[gold] += 1
        if judge(gold) != 0 and judge(guess) != 0:
            guess_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[gold] += 1

    f1_by_relation = Counter()
    recall_by_relation = Counter()
    prec_by_relation = Counter()
    for rel in relations:
        # 不处理no_relation的情况
        if judge(rel) == 0:
            continue
        recall = 0
        if gold_by_relation[rel] > 0:
            recall = correct_by_relation[rel] / gold_by_relation[rel]
        precision = 0
        if guess_by_relation[rel] > 0:
            precision = correct_by_relation[rel] / guess_by_relation[rel]
        if recall + precision > 0 :
            f1_by_relation[rel] = 2 * recall * precision / (recall + precision)
        recall_by_relation[rel] = recall
        prec_by_relation[rel] = precision

    micro_f1 = 0
    prec = 0
    recall = 0
    if sum(guess_by_relation.values()) != 0 and sum(correct_by_relation.values()) != 0:
        recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
        prec = sum(correct_by_relation.values()) / sum(guess_by_relation.values())    
        micro_f1 = 2 * recall * prec / (recall+prec)

    return micro_f1, prec, recall, f1_by_relation, prec_by_relation, recall_by_relation

def precision_recall_fscore_(labels, preds, n_labels=42):
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=list(range(1, n_labels)), average="micro")
    return p, r, f


def apply_threshold(output, threshold=0.0, ignore_negative_prediction=True):
    """Applies a threshold to determine whether is a relation or not"""
    output_ = output.copy()
    # print("output_")
    # print(output_.shape)
    if ignore_negative_prediction:
        output_[:, 0] = 0.0
    activations = (output_ >= threshold).sum(-1).astype(np.int)
    # print("activations")
    # print(activations)
    # print(activations.shape)
    output_[activations == 0, 0] = 1.00

    return output_.argmax(-1)


def find_optimal_threshold(labels, output, granularity=1000, metric=accuracy_):
    thresholds = np.linspace(0, 1, granularity)
    values = []
    for t in thresholds:
        preds = apply_threshold(output, threshold=t)
        values.append(metric(labels, preds))

    best_metric_id = np.argmax(values)
    best_threshold = thresholds[best_metric_id]

    return best_threshold, values[best_metric_id]
