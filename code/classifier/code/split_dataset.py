import os, json, random

path_1 = "/home/lypl/trldc-main/data/CAMS_json_data_0.7/5/filted/IntentSDCNL_Deving.json"
path_2 = "/home/lypl/trldc-main/data/CAMS_json_data_0.7/5/filted/IntentSDCNL_Training.json"
all_train_data = {}
with open(path_1, 'r', encoding='ISO-8859-1') as f:
    for line in f.readlines():
        line = json.loads(line)
        if line['labels'][0] not in all_train_data.keys():
            all_train_data[line['labels'][0]] = []
        all_train_data[line['labels'][0]].append(line)
with open(path_2, 'r', encoding='ISO-8859-1') as f:
    for line in f.readlines():
        line = json.loads(line)
        if line['labels'][0] not in all_train_data.keys():
            all_train_data[line['labels'][0]] = []
        all_train_data[line['labels'][0]].append(line)

prop = 0.7
out_dir = "/home/lypl/trldc-main/data/CAMS_json_data_0.7"
for idx in range(6, 10):
    train = []
    dev = []
    for key in all_train_data.keys():
        random.shuffle(all_train_data[key])
        train_num = int(prop*len(all_train_data[key]))
        train.extend(all_train_data[key][:train_num])
        dev.extend(all_train_data[key][train_num:])
    random.shuffle(train)
    random.shuffle(dev)
    out_train_path = os.path.join(out_dir, str(idx), "filted", "IntentSDCNL_Training.json")
    out_dev_path = os.path.join(out_dir, str(idx), "filted", "IntentSDCNL_Deving.json")
    with open(out_train_path, 'w', encoding="ISO-8859-1") as f:
        for dt in train:
            f.write(json.dumps(dt))
            f.write("\n")
    with open(out_dev_path, 'w', encoding="ISO-8859-1") as f:
        for dt in dev:
            f.write(json.dumps(dt))
            f.write("\n")
