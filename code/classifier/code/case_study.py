import os, json, math
m_b = 56.6439927
m_a = 61.6045549
n_b = 4.0
n_a = 5.0
s_b = 3.48105947
s_a = 1.540160317
t_value = (m_a-m_b) / (math.sqrt((s_a*s_a/n_a)+(s_b*s_b/n_b)))
print(t_value)
# ori_test = []
# label = []
# text = []
# with open("/home/lypl/trldc-main/data/CAMS_json_data_0.7/11/filted/IntentSDCNL_Testing.json", 'r', encoding='ISO-8859-1') as f:
#     for line in f.readlines():
#         line = json.loads(line)
#         ori_test.append(line)
#         label.append(line["labels"])
#         text.append(line["text"])

# with open("/media/data/3/lyp/case_study/CAMS_length_4096_seed_42_filted.json", 'r') as f:
#     filted_res = json.load(f)

# with open("/media/data/3/lyp/case_study/CAMS_length_4096_seed_42.json", 'r') as f:
#     ori_res = json.load(f)

# for i in range(len(ori_res)):
#     now_ori_res = ori_res[i]
#     now_filted_res = filted_res[i]
#     now_label = label[i]
#     now_text = text[i]
#     now_len = len(text[i].split())
#     if now_len <= 100:
#         print(now_label)
#         print(now_ori_res)
#         print(now_filted_res)
#         print(now_text)
#         input("------")