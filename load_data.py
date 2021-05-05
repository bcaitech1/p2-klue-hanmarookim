import pickle as pickle
import os
import pandas as pd
import torch
from pororo import Pororo


# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, label):
        self.tokenized_dataset = tokenized_dataset
        self.label = label

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.label[idx])
        return item

    def __len__(self):
        return len(self.label)


# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type):
    label = []
    for i in dataset[8]:
        if i == 'blind':
            label.append(100)
        else:
            label.append(label_type[i])
    out_dataset = pd.DataFrame(
        {'sentence': dataset[1], 'entity_01': dataset[2], 'entity_02': dataset[5], 'label': label, })
    return out_dataset


def ner_preprocessing_dataset(dataset, label_type):
    label = list(map(int, dataset[4][1:]))
    out_dataset = pd.DataFrame(
        {'sentence': dataset[1][1:], 'entity_01': dataset[2][1:], 'entity_02': dataset[3][1:], 'label': label, })
    return out_dataset


def entity_token_preprocessing_dataset(dataset, label_type):
    label = []
    e1_s = dataset[3]
    e1_e = dataset[4]
    e2_s = dataset[6]
    e2_e = dataset[7]
    sentence = []
    for d in range(len(dataset[1])):
        idxs = zip([1,1,2,2], [e1_s[d], e1_e[d], e2_s[d], e2_e[d]])
        idxs = sorted(idxs, key=lambda x: x[1])
        sentence.append(dataset[1][d][:idxs[0][1]]+f'[E{idxs[0][0]}]'+dataset[1][d][idxs[0][1]:idxs[1][1]+1]+f'[/E{idxs[1][0]}]'+dataset[1][d][idxs[1][1]+1:idxs[2][1]]+f'[E{idxs[2][0]}]'+dataset[1][d][idxs[2][1]:idxs[3][1]+1]+f'[/E{idxs[3][0]}]'+dataset[1][d][idxs[3][1]+1:])

    for i in dataset[8]:
        if i == 'blind':
            label.append(100)
        else:
            label.append(label_type[i])
    out_dataset = pd.DataFrame(
        {'sentence': sentence, 'e1_s': dataset[3], 'e1_e': dataset[4],  'e2_s': dataset[6], 'e2_e':dataset[7], 'label': label, })
    return out_dataset

# tsv 파일을 불러옵니다.
def load_data(dataset_dir):
    # load label_type, classes
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    # load dataset
    dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
    # preprecessing dataset
    dataset = preprocessing_dataset(dataset, label_type)

    return dataset

def ner_load_data(dataset_dir):
    # load label_type, classes
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    # load dataset
    dataset = pd.read_csv(dataset_dir, header=None)
    # preprecessing dataset
    dataset = ner_preprocessing_dataset(dataset, label_type)

    return dataset



def entity_token_load_data(dataset_dir):
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    # load dataset
    dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
    # preprecessing dataset
    dataset = entity_token_preprocessing_dataset(dataset, label_type)

    return dataset


# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
def tokenized_dataset(dataset, tokenizer):
    concat_entity = []
    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=100,
        add_special_tokens=True,
    )
    return tokenized_sentences

def ner_tokenized_dataset(dataset, tokenizer):
    ner = Pororo(task='ner', lang='ko')
    sentence = list(dataset['sentence'])
    concat_entity = []
    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
        temp = '[E1]'+'[E1-NER]'+ner(e01)[0][1]+'[E1-NER]'+e01+'[E1]'+'[E2]'+'[E2-NER]'+ner(e02)[0][1]+'[E2-NER]'+e02+'[E2]'
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=200,
        add_special_tokens=True,

    )
    return tokenized_sentences

def single_tokenized_dataset(dataset, tokenizer):
    sentence = list(dataset['sentence'])
    tokenized_sentences = tokenizer(
        sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=150,
        add_special_tokens=True,

    )
    return tokenized_sentences

two_sentence_template = '{}와 {}은 무슨 관계인가?'


def two_sentence_tokenized_dataset(dataset, tokenizer):
    concat_entity = []
    modified = [s for s in dataset['sentence'] for x in range(2)]
    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
        temp1 = two_sentence_template.format(e01, e02)
        temp2 = two_sentence_template.format(e02, e01)
        concat_entity.append(temp1)
        concat_entity.append(temp2)
    tokenized_sentences = tokenizer(
        modified,
        concat_entity,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=150,
        add_special_tokens=True,
    )
    return tokenized_sentences





class TwoSentenceDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, label):
        self.tokenized_dataset = tokenized_dataset
        self.label = label

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
        item['label'] = torch.tensor(self.label[idx//2])
        return item

    def __len__(self):
        return len(self.label)*2


rtq_template = ('{}의 {}은 {}인가?', '{}은 {} {}의 {}인가?')


def rtq_tokenized_dataset(dataset, tokenizer):
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    label_type = {label_type[value]:value for value in label_type.keys()}
    concat_entity = []
    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
        for k in label_type.keys():
            if k == 0: continue
            s = label_type[k].split(':')
            temp1 = rtq_template[0].format(e01, s[1], e02)
            temp2 = rtq_template[1].format(e02, s[0], e01, s[1])
            concat_entity.append(temp1)
            concat_entity.append(temp2)
    modified = [s for s in dataset['sentence'] for x in range(82)]
    tokenized_sentences = tokenizer(
        modified,
        concat_entity,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=200,
        add_special_tokens=True,
    )
    return tokenized_sentences


class RtQDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, label):
        self.tokenized_dataset = tokenized_dataset
        self.label = label

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
        if self.label[idx//82] == 0:
            item['label'] = torch.tensor(0)
        else:
            label = self.label[idx//82]
            if idx%82 == (label-1)*2 or idx%82 == ((label-1)*2)+1:
                item['label'] = torch.tensor(1)
            else:
                item['label'] = torch.tensor(0)
        return item

    def __len__(self):
        return len(self.label) * 82
