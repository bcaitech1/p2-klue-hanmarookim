from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, \
    BertTokenizer
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers import AutoModelForSequenceClassification
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
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)
        output_pred.append(result)

    return np.array(output_pred).flatten()


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
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)
        output_pred.append(result)

    return np.array(output_pred).flatten()

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


def rtq_load_test_dataset(dataset_dir, tokenizer):
    test_dataset = load_data(dataset_dir)
    test_label = test_dataset['label'].values
    tokenized_test = rtq_tokenized_dataset(test_dataset, tokenizer)
    return tokenized_test, test_label


def two_sentence_test_dataset(dataset_dir, tokenizer):
    test_dataset = load_data(dataset_dir)
    test_label = test_dataset['label'].values
    tokenized_test = two_sentence_tokenized_dataset(test_dataset, tokenizer)
    return tokenized_test, test_label


def rtq_inference(model, tokenized_sent, tokenizer, device):
    dataloader = DataLoader(tokenized_sent, batch_size=82, shuffle=False)
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
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)
        print(i, tokenizer.decode(data['input_ids'][0]), result[0])
        output_pred.append(result)

    return np.array(output_pred).flatten()


def two_sentence_inference(model, tokenized_sent, tokenizer, device):
    dataloader = DataLoader(tokenized_sent, batch_size=32, shuffle=False)
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
        logits_sf = F.softmax(logits, dim=1)
        logits = logits.detach().cpu().numpy()
        result = []
        for i in range(0, len(logits_sf), 2):
            softvoting = ((logits_sf[i] + logits_sf[i+1])/2).detach().cpu().numpy()
            output_pred.append(np.argmax(softvoting))

    return np.array(output_pred).flatten()


def except0_inference(model, tokenized_sent, device):
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
        logits_sigmoid = torch.sigmoid(logits)
        logits = logits.detach().cpu().numpy()
        result = []
        pred = np.argmax(logits, axis=-1)
        for i, idx in enumerate(pred):
            if logits_sigmoid[i, idx] < 0.6:
                result.append(0)
            else:
                result.append(idx+1)
        output_pred.append(result)

    return np.array(output_pred).flatten()


def main(model_dir, args):
    """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load tokenizer
    if 'electra' in args['MODEL_NAME']:
        TOK_NAME = args['MODEL_NAME']
        if args['entity_token'] == 'on':
            tokenizer = ElectraTokenizer.from_pretrained(TOK_NAME, additional_special_tokens=['[E1]', '[E2]', '[E1-NER]', '[E2-NER]'])
        else:
            tokenizer = ElectraTokenizer.from_pretrained(TOK_NAME)


    else:
        TOK_NAME = args['MODEL_NAME']
        tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)
        if args['entity_token'] == 'on':
            tokenizer = AutoTokenizer.from_pretrained(TOK_NAME, additional_special_tokens=['[E1]', '[E2]', '[E1-NER]', '[E2-NER]'])

    # load my model
    MODEL_NAME = model_dir  # model dir.
    if 'electra' in args['MODEL_NAME']:
        model = ElectraForSequenceClassification.from_pretrained(model_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    if args['entity_token'] == 'on':
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # load test datset
    test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
    if args['rtq'] == 'on':
        test_dataset, test_label = rtq_load_test_dataset(test_dataset_dir, tokenizer)
        #print(len(test_dataset['input_ids']))
        test_dataset = RtQDataset(test_dataset, test_label)
    elif args['two_sentence'] == 'on':
        test_dataset, test_label = two_sentence_test_dataset(test_dataset_dir, tokenizer)
        test_dataset = TwoSentenceDataset(test_dataset, test_label)
    else:
        if args['entity_token'] == 'on':
            test_dataset, test_label = ner_load_test_dataset('/opt/ml/input/data/test/ner_test_normalize.csv', tokenizer)

            #test_dataset = ner_load_data(test_dataset_dir)
            #test_label = test_dataset['label'].values
            # tokenizing dataset
            #test_dataset = single_tokenized_dataset(test_dataset, tokenizer)
        else:
            test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
        test_dataset = RE_Dataset(test_dataset, test_label)
    print(test_dataset[0])

    # predict answer
    if args['rtq'] == 'on':
        pred_answer = rtq_inference(model, test_dataset, tokenizer, device)
    if args['two_sentence'] == 'on':
        pred_answer = two_sentence_inference(model, test_dataset, tokenizer, device)
    if args['except_0'] == 'on':
        pred_answer = except0_inference(model, test_dataset, device)
    else:
        if 'xlm' in args['MODEL_NAME']:
            pred_answer = xlm_inference(model, test_dataset, device)
        else:
            pred_answer = inference(model, test_dataset, device)
    # make csv file with predicted answer
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.

    output = pd.DataFrame(pred_answer, columns=['pred'])
    s = args['logging_dir'].split('/')
    output.to_csv('./prediction/{}_2.csv'.format(s[-1]), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model dir
    parser.add_argument('--model_dir', type=str, default="./results/checkpoint-500")
    parser.add_argument('--config_file', type=str, required=True)
    args = parser.parse_args()
    model_dir = args.model_dir
    with open(args.config_file) as f:
        args = AttrDict(json.load(f))
    print(args)
    main(model_dir, args)

