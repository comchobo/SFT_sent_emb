from sentence_transformers import InputExample


def group_by_column(dic):
    res = []
    grouped_data = {}
    for premise in dic['premise']:
        grouped_data[premise] = {}

    for premise, hypothesis, label in zip(dic['premise'], dic['hypothesis'], dic['label']):
        if label == 1 : continue
        else: grouped_data[premise][label] = hypothesis
    for item in grouped_data.items():
        try:
            res.append([item[0], item[1][0], item[1][2]])
        except KeyError:
            continue
    return res


def preprocess_nli(nli_dataset):
    train_res = []
    valid_res = []

    train_grouped = group_by_column(nli_dataset['train'])
    valid_grouped = group_by_column(nli_dataset['validation'])
    for row in train_grouped:
        train_res.append(InputExample(texts=[row[0], row[1], row[2]]))
    for row in valid_grouped:
        valid_res.append(InputExample(texts=[row[0], row[1], row[2]]))
    return train_res, valid_res


def preprocess_sts(sts_dataset):
    train_res=[]
    valid_res=[]
    for row in sts_dataset['train']:
        train_res.append(InputExample(texts=[row['sentence1'], row['sentence2']],
                                      label= float(row['labels']['label'])/5.0))
    for row in sts_dataset['validation']:
        valid_res.append(InputExample(texts=[row['sentence1'], row['sentence2']],
                                      label= float(row['labels']['label'])/5.0))
    return train_res, valid_res
