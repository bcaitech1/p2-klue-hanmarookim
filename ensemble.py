from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, \
    BertTokenizer
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.utils.data import DataLoader
import torch.nn.functional as F
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
import json
from attrdict import AttrDict
def inference(model, tokenized_sent, device):
    dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
    model.eval()
    output_pred = []

    for i, data in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                token_type_ids=data['token_type_ids'].to(device)
            )
        logits = outputs[0]
        result = F.softmax(logits, dim=1)
        result = result.detach().cpu().numpy()
        output_pred.extend(result)

    return np.array(output_pred)


def xlm_inference(model, tokenized_sent, device):
    dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
    model.eval()
    output_pred = []

    for i, data in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
            )
        logits = outputs[0]
        result = F.softmax(logits, dim=1)
        result = result.detach().cpu().numpy()

        output_pred.extend(result)

    return np.array(output_pred)

def load_test_dataset(dataset_dir, tokenizer):
    test_dataset = load_data(dataset_dir)
    test_label = test_dataset['label'].values
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return tokenized_test, test_label

def ner_load_test_dataset(dataset_dir, tokenizer):
    test_dataset = ner_load_data(dataset_dir)
    test_label = test_dataset['label'].values
    # tokenizing dataset
    tokenized_test = ner_tokenized_dataset(test_dataset, tokenizer)
    return tokenized_test, test_label

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model1_tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
    model1 = ElectraForSequenceClassification.from_pretrained('/opt/ml/pycharm/outputs/conf3_koelectra_base_v3_pretrained/checkpoint-8445/')
    model1.to(device)
    model2_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
    model2 = XLMRobertaForSequenceClassification.from_pretrained('/opt/ml/pycharm/outputs/conf18_xlm-roberta-large_discussion/checkpoint-2820/')
    model2.to(device)
    model3_tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator', additional_special_tokens=['[E1]', '[E2]', '[E1-NER]', '[E2-NER]'])
    model3 = ElectraForSequenceClassification.from_pretrained(
        '/opt/ml/pycharm/outputs/conf22_conf3_koelectra_base_v3/checkpoint-8445/')
    model3.to(device)
    model4_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large', additional_special_tokens=['[E1]', '[E2]', '[E1-NER]', '[E2-NER]'])
    model4 = XLMRobertaForSequenceClassification.from_pretrained(
        '/opt/ml/pycharm/outputs/conf21_xlm-roberta-large_discussion_ner_entity_normalize/checkpoint-2820/')
    model4.to(device)

    test_dataset_dir = '/opt/ml/input/data/test/test.tsv'
    test_dataset1, test_label = load_test_dataset(test_dataset_dir, model1_tokenizer)
    test_dataset2, test_label = load_test_dataset(test_dataset_dir, model2_tokenizer)
    test_dataset_ner = '/opt/ml/input/data/test/ner_test_normalize.csv'
    test_dataset3, test_label = ner_load_test_dataset(test_dataset_ner, model3_tokenizer)
    test_dataset4, test_label = ner_load_test_dataset(test_dataset_ner, model4_tokenizer)
    test_dataset1 = RE_Dataset(test_dataset1, test_label)
    test_dataset2 = RE_Dataset(test_dataset2, test_label)
    test_dataset3 = RE_Dataset(test_dataset3, test_label)
    test_dataset4 = RE_Dataset(test_dataset4, test_label)

    result1 = inference(model1, test_dataset1, device)
    result2 = xlm_inference(model2, test_dataset2, device)
    result3 = inference(model3, test_dataset3, device)
    result4 = xlm_inference(model4, test_dataset4, device)
    pred = np.argmax((result1+result2+result3+result4)/4, axis=-1)
    output = pd.DataFrame(pred, columns=['pred'])

    print(test_dataset1[0]['input_ids'])
    print(test_dataset2[0]['input_ids'])
    print(test_dataset3[0]['input_ids'])
    print(test_dataset4[0]['input_ids'])
    output.to_csv('./prediction/ensemble2.csv', index=False)

if __name__ == '__main__':
    main()