import os, json, random, math, sys
import torch
import numpy as np
import tqdm

LABEL_WORDS = {
    '0': ["no reason"],
    '1': ['body shame', 'physical abuse', 'sexual abuse', 'emotional abuse'],
    '2': ['loans', 'poverty', 'financial problem', 'unemployment', 'bully', ' gossip', 'bad job', 'poor grades', 'educational problem', 'dictatorial management'],
    '3': ['drug', 'medication', 'tumor', 'cancer', 'disease', 'alcohol addiction', 'smoke'],
    '4': ['bad parenting', 'breakup', 'divorce', 'mistrust', 'jealousy', 'betrayal', 'conflict', 'fight', 'childhood trauma'],
    '5': ['worthless', 'lonely', 'tired  of daily', 'powerless', 'normless', ' isolation', 'estrangement'],
}




def get_config(verbalizer):
    ret = [{
        "name": "roberta-large-mnli",
        "classification_model": "mnli-mapping",
        "pretrained_model": "/media/data/3/lyp/roberta-large-mnli",
        "batch_size": 4,
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
        "template_mapping": verbalizer,
    }]
    return ret

def get_verbalizer(mode, template):
    ret = {}
    if mode == 'co-share':
        for clss in LABEL_WORDS.keys():
            ret[clss] = []
            for lw in LABEL_WORDS[clss]:
                ret[clss].append(template.format(label_words=lw))
    else:
        pass
    return ret

tmp_config_path = "/home/lypl/social_stc_score/a2t/relation_classification/configs/zero-shot/config_for_boosting_train.json"


template_num = int(sys.argv[1])
print("template_num")
print(template_num)
in_dir = os.path.join("/media/data/3/lyp/stc_score_ckpt/boosting", "template_num", str(template_num))

learner_weights = np.load(os.path.join(in_dir, "learner_weight.npy"))
weak_learner_path = os.path.join(in_dir, "weak_learners.json")

with open(weak_learner_path, 'r') as f:
    weak_learners = json.load(f)


test_1_path = "/media/data/3/lyp/CAM_stc_score_dataset/train_no_key_stc_nli.json"
test_2_path = "/media/data/3/lyp/CAM_stc_score_dataset/dev_no_key_stc_nli.json"
with open(test_1_path, 'r', encoding='ISO-8859-1') as f:
    test_1 = json.load(f)
with open(test_2_path, 'r', encoding='ISO-8859-1') as f:
    test_2 = json.load(f)
test_1_prob = np.zeros((len(test_1), 6))
test_2_prob = np.zeros((len(test_2), 6))


for t_idx, t in enumerate(weak_learners):
    # verbalizer = t
    # 当前verbalizer用打分器给所有训练集 在5个类别上打分
    config = get_config(t)
    # 写入要运行的config
    with open(tmp_config_path, 'w') as f:
        json.dump(config, f)
    # 运行
    gpu = sys.argv[2]
    cmd = "CUDA_VISIBLE_DEVICES="+str(gpu)
    os.system(cmd+" python3 -m a2t.relation_classification.run_evaluation \
    /media/data/3/lyp/CAM_stc_score_dataset/train_no_key_stc_nli.json \
    --config /home/lypl/social_stc_score/a2t/relation_classification/configs/zero-shot/config_for_boosting_train.json")
    
    # 读取out_put
    
    test_1_out_puts = np.load(os.path.join("/home/lypl/social_stc_score/a2t/relation_classification/boosting", config[0]['name'], "output_train.npy"))
    # 每个类别上
    for i in range(test_1_out_puts.shape[1]):
        for dt_idx in range(test_1_out_puts.shape[0]):
            test_1_prob[dt_idx, i] += learner_weights[t_idx] * test_1_out_puts[dt_idx, i]

        

    cmd = "CUDA_VISIBLE_DEVICES="+str(gpu)
    os.system(cmd+" python3 -m a2t.relation_classification.run_evaluation \
    /media/data/3/lyp/CAM_stc_score_dataset/dev_no_key_stc_nli.json \
    --config /home/lypl/social_stc_score/a2t/relation_classification/configs/zero-shot/config_for_boosting_train.json")
    
    # 读取out_put
    
    test_2_out_puts = np.load(os.path.join("/home/lypl/social_stc_score/a2t/relation_classification/boosting", config[0]['name'], "output_dev.npy"))
    # 每个类别上
    for i in range(test_2_out_puts.shape[1]):
        for dt_idx in range(test_2_out_puts.shape[0]):
            test_2_prob[dt_idx, i] += learner_weights[t_idx] * test_2_out_puts[dt_idx, i]

for dt_idx in range(test_1_out_puts.shape[0]):
    test_1_res = np.argmax(test_1_prob, axis=1)
    print(test_1_res.shape)


for dt_idx in range(test_2_out_puts.shape[0]):
    test_2_res = np.argmax(test_2_prob, axis=1)
    print(test_1_res.shape)

out_dir = os.path.join("/home/lypl/social_stc_score/a2t/relation_classification/experiments/all_no_key_stc",str(template_num))
os.makedirs(out_dir, exist_ok=True)
out_path_1 = os.path.join(out_dir, "output_train.npy")
out_path_2 = os.path.join(out_dir, "output_dev.npy")
np.save(out_path_1, np.array(test_1_res))
np.save(out_path_2, np.array(test_2_res))