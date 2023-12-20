''' test data(for nli-pre version) to nli-test version '''
import os
import json

def info_to_dict(text_id, stc, label, annotation):
    ret = {
        "text_id": text_id,
        "token": stc,
        "relation": annotation,
        "is_key": label
    }
    return ret

''' test '''
in_path = "/media/data/3/lyp/CAM_stc_score_dataset/test.json"
with open(in_path, 'r', encoding='ISO-8859-1') as f:
    raw_data = json.load(f)

data = []
for i, dt in enumerate(raw_data):
    if '0' in dt['annotation']:
        # for it, stc in enumerate(dt['sentences']):
        #     data.append(info_to_dict(i, stc, 0, '0')) # 因为0类的句子不能推断出任何原因，所以不是key
        continue # test只需要选取positive example的结果进行model selection
    else:
        for it, stc in enumerate(dt['sentences']):
            data.append(info_to_dict(i, stc, dt['labels'][it], dt['annotation'][0]))

out_path = "/media/data/3/lyp/CAM_stc_score_dataset/test_nli.json"
with open(out_path, 'w', encoding='ISO-8859-1') as f:
    json.dump(data, f)



''' train_no_key_stc  '''
in_path = "/media/data/3/lyp/CAM_stc_score_dataset/train_no_key_stc.json"
with open(in_path, 'r', encoding='ISO-8859-1') as f:
    raw_data = json.load(f)

data = []
for i, dt in enumerate(raw_data):
    if '0' in dt['annotation']:
        # for it, stc in enumerate(dt['sentences']):
        #     data.append(info_to_dict(i, stc, 0, '0')) # 因为0类的句子不能推断出任何原因，所以不是key
        continue # 不能推理出0类的key stc，不需要进行预测
    else:
        for it, stc in enumerate(dt['sentences']):
            data.append(info_to_dict(i, stc, -1, dt['annotation'][0]))
            # if it >= len(dt['labels']):
            #     continue
            # else: # ⭐这里写错了，train_no_key_stc里没有len(dt['labels']) > 0的样本，所以只跑NLI得到output_即可，不计算accuracy
            #     data.append(info_to_dict(i, stc, dt['labels'][it], dt['annotation'][0]))

out_path = "/media/data/3/lyp/CAM_stc_score_dataset/train_no_key_stc_nli.json"
with open(out_path, 'w', encoding='ISO-8859-1') as f:
    json.dump(data, f)


''' dev_no_key_stc '''
in_path = "/media/data/3/lyp/CAM_stc_score_dataset/dev_no_key_stc.json"
with open(in_path, 'r', encoding='ISO-8859-1') as f:
    raw_data = json.load(f)

data = []
for i, dt in enumerate(raw_data):
    if '0' in dt['annotation']:
        # for it, stc in enumerate(dt['sentences']):
        #     data.append(info_to_dict(i, stc, 0, '0')) # 因为0类的句子不能推断出任何原因，所以不是key
        continue # 不能推理出0类的结果
    else:
        for it, stc in enumerate(dt['sentences']):
            data.append(info_to_dict(i, stc, -1, dt['annotation'][0]))
            # if it >= len(dt['labels']):
            #     ''' ⭐1-5类，但是没有明显的解释，需要判断多个句子才有解释，所以能推断出原因，但是需要整个文章一起推理,这里不用于构造数据集 '''
            #     continue
            # else:
            #     ''' 1-5类，有明显的解释 '''
            #     data.append(info_to_dict(i, stc, dt['labels'][it], dt['annotation'][0]))
# print(cnt)

out_path = "/media/data/3/lyp/CAM_stc_score_dataset/dev_no_key_stc_nli.json"
with open(out_path, 'w', encoding='ISO-8859-1') as f:
    json.dump(data, f)


''' train_have_key_stc  '''
in_path = "/media/data/3/lyp/CAM_stc_score_dataset/train_have_key_stc.json"
with open(in_path, 'r', encoding='ISO-8859-1') as f:
    raw_data = json.load(f)

data = []
for i, dt in enumerate(raw_data):
    if '0' in dt['annotation']:
        # for it, stc in enumerate(dt['sentences']):
        #     data.append(info_to_dict(i, stc, 0, '0')) # 因为0类的句子不能推断出任何原因，所以不是key
        continue # 不能推理出0类的key stc，不需要进行预测
    else:
        for it, stc in enumerate(dt['sentences']):
            data.append(info_to_dict(i, stc, dt['labels'][it], dt['annotation'][0]))
            # if it >= len(dt['labels']):
            #     continue
            # else: # ⭐这里写错了，train_no_key_stc里没有len(dt['labels']) > 0的样本，所以只跑NLI得到output_即可，不计算accuracy
            #     data.append(info_to_dict(i, stc, dt['labels'][it], dt['annotation'][0]))

out_path = "/media/data/3/lyp/CAM_stc_score_dataset/train_have_key_stc_nli.json"
with open(out_path, 'w', encoding='ISO-8859-1') as f:
    json.dump(data, f)


''' dev_have_key_stc '''
in_path = "/media/data/3/lyp/CAM_stc_score_dataset/dev_have_key_stc.json"
with open(in_path, 'r', encoding='ISO-8859-1') as f:
    raw_data = json.load(f)

data = []
for i, dt in enumerate(raw_data):
    if '0' in dt['annotation']:
        # for it, stc in enumerate(dt['sentences']):
        #     data.append(info_to_dict(i, stc, 0, '0')) # 因为0类的句子不能推断出任何原因，所以不是key
        continue # 不能推理出0类的结果
    else:
        for it, stc in enumerate(dt['sentences']):
            data.append(info_to_dict(i, stc, dt['labels'][it], dt['annotation'][0]))
            # if it >= len(dt['labels']):
            #     ''' ⭐1-5类，但是没有明显的解释，需要判断多个句子才有解释，所以能推断出原因，但是需要整个文章一起推理,这里不用于构造数据集 '''
            #     continue
            # else:
            #     ''' 1-5类，有明显的解释 '''
            #     data.append(info_to_dict(i, stc, dt['labels'][it], dt['annotation'][0]))
# print(cnt)

out_path = "/media/data/3/lyp/CAM_stc_score_dataset/dev_have_key_stc_nli.json"
with open(out_path, 'w', encoding='ISO-8859-1') as f:
    json.dump(data, f)