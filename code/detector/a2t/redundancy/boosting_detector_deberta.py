import os, json, random, math
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
        "name": "deberta-large-mnli",
        "classification_model": "mnli-mapping",
        "pretrained_model": "/media/data/3/lyp/deberta-large-mnli",
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

def get_tpl_from_file(file):
    span_list = [file['template'][1]["content"].strip()]
    span_list.append("{label_words}")
    span_list.append(file['template'][3]["content"].strip())
    return ' '.join(span_list)
''' 获取所有tpl set '''

templates_list = []
templates_dir = os.path.join("/home/lypl/PromptBoosting/templates/full_templates/t5_sorted_social")

file_list = os.listdir(templates_dir)
for file_name in file_list:
    path = os.path.join(templates_dir, file_name)
    with open(path, 'r') as f:
        tpl_file = json.load(f)
        tpl = get_tpl_from_file(tpl_file)
        templates_list.append(tpl)


learner_num = len(templates_list)
weak_learner = []
learner_weights = [] # weight in numpy
# ⭐ 这里11007是train+have_key_stc的数据个数
data_weights = np.full((11007,1),float(1.0/float(learner_num)),dtype=np.float16)
tmp_config_path = "/home/lypl/social_stc_score/a2t/relation_classification/configs/zero-shot/config_for_boosting_train_deberta.json"
tmp_out_puts_dir = os.path.join("/home/lypl/social_stc_score/a2t/relation_classification/boosting")

def get_error_t(pred, label):
    # assert(pred.shape[0] == len(label))
    sum = 0.0
    for i in range(pred.shape[0]):
        if pred[i] != label[i]:
            sum += data_weights[i, 0]
    return sum/data_weights.sum()


def update_data_weight(data_weights, alpha_t, pred, label):
    # nmupy 对应位置相乘
    tmp = []
    for i in range(pred.shape[0]):
        if pred[i] != label[i]:
            tmp.append(1.0)
        else:
            tmp.append(0.0)
    weight_t = np.array(tmp)
    weight_t = weight_t * alpha_t
    data_weights = data_weights*weight_t
    return data_weights

''' 准备训练数据集, 保持顺序,不用分dev集，直接把所有train中有interpretation的数据处理好弄上去 '''
is_key_labels = []
with open("/home/lypl/social_stc_score/a2t/relation_classification/train_have_key_stc_nli.json", 'r') as f:
    raw_data = json.load(f)
for line in raw_data:
    is_key_labels.append(line["is_key"])

def cal_learner_t_weight(error_t):
    return math.log( (1.0-error_t)/error_t ) + math.log(5)

''' boosting算法 '''
for t in tqdm.tqdm(range(learner_num)):
    # 两种方式通过template获得一组verbalizer ['co-share', 'respectively']
    verbalizer = get_verbalizer("co-share", templates_list[t])
    weak_learner.append(verbalizer)
    # 当前verbalizer用打分器给所有训练集 在5个类别上打分
    config = get_config(verbalizer)
    # 写入要运行的config
    with open(tmp_config_path, 'w') as f:
        json.dump(config, f)
    # 运行
    os.system("CUDA_VISIBLE_DEVICES=7 python3 -m a2t.relation_classification.run_evaluation \
    /home/lypl/social_stc_score/a2t/relation_classification/train_have_key_stc_nli.json \
    --config /home/lypl/social_stc_score/a2t/relation_classification/configs/zero-shot/config_for_boosting_train_deberta.json")
    
    # 读取out_put
    
    out_puts = np.load(os.path.join("/home/lypl/social_stc_score/a2t/relation_classification/experiments/", config[0]['name'], "output_train.npy"))
    # 获取error_t
    error_t = get_error_t(out_puts, is_key_labels)
    # 计算learner_t的权重
    alpha_t = cal_learner_t_weight(error_t)
    # print(t, alpha_t)
    learner_weights.append(alpha_t)
    # 更新训练集权重
    data_weights = update_data_weight(data_weights, alpha_t, out_puts, is_key_labels)
    # print(data_weights)

np.save("/media/data/3/lyp/stc_score_ckpt/boosting/co-share/deberta/data_weight.npy", data_weights)
np.save("/media/data/3/lyp/stc_score_ckpt/boosting/co-share/deberta/learner_weight.npy", np.array(learner_weights))
weak_learner_path = "/media/data/3/lyp/stc_score_ckpt/boosting/co-share/deberta/weak_learners.json"
with open(weak_learner_path, 'w') as f:
    json.dump(weak_learner, f, indent=4)


''' 根据每个weak learner计算最后的结果，每个test数据的0/1结果是多个weak learner的加权平均后的结果(soft),最后预测成0就删去，1就留下（这里写公式到算法里） '''
# train_path = "/media/data/3/lyp/CAM_stc_score_dataset/train_no_key_stc_nli.json"
# dev_path = "/media/data/3/lyp/CAM_stc_score_dataset/dev_no_key_stc_nli.json"
# with open(train_path, 'r', encoding='ISO-8859-1') as f:
#     train_no_key_stc = json.load(f)
# with open(dev_path, 'r', encoding='ISO-8859-1') as f:
#     dev_no_key_stc = json.load(f)

# train_weighted_avg = []
# dev_weighted_avg = []
# ''' train '''
# for t in range(learner_num):
#     test_config = get_config(weak_learner[t])
#     with open(tmp_config_path, 'w') as f:
#         json.dump(test_config)
#     os.system()
# now_dir = (tmp_out_puts_dir, config[0]["name"])
#     os.makedirs(now_dir, exist_ok=True)
#     tmp_out_puts_path = os.path.join(now_dir, "output_train.npy")
#     out_puts = np.load(tmp_out_puts_path) # 每个类别（[0, 5]）上的概率 
#     for i in range(outputs.shape[0]):
#         train_weighted_avg.append()
    
# out_path = /"home/lypl/social_stc_score/a2t/relation_classification/experiments/roberta-large-mnli/output_train.npy"
# np.save(out_path, np.array(to_is_key(train_weighted_avg)))

# ''' dev '''


