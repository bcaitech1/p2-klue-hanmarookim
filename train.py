import pickle as pickle
import os
import pandas as pd
import torch
import argparse
import glob
import json
import time
import numpy as np
import random
from attrdict import AttrDict

from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import ElectraTokenizer, ElectraForSequenceClassification, ElectraConfig
from transformers import XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback
from load_data import *


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# 평가를 위한 metrics function.
def compute_metrics(pred):
    label = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(label, preds)
    return {
        'accuracy': acc,
    }

"""
class testTrainer(Trainer):
    def __init__(self):
        self.criterion = torch.nn.BCEWithLogitsLoss()
    def compute_loss(self, model, inputs, return_outputs=False):
        label = inputs.pop('label')
        test_targets = torch.zeros((len(label), 41))
        for l in range(len(label)):
            if label[l] == 0:
                test_targets[l, :] = 1/41
            else:
                idx = label[l]-1
                test_targets[l, idx] = 1
        test_outputs = model(**inputs)
        print(test_outputs)

        loss = self.criterion(test_outputs, test_targets)
        print(test_targets)
        exit(0)
        return (loss, test_outputs) if return_outputs else loss
"""

class testTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        label = inputs.pop("label")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        test_targets = torch.zeros((len(label), 41), device='cuda:0')
        for l in range(len(label)):
            if label[l] == 0:
                test_targets[l, :] = 1 / 41
            else:
                idx = label[l] - 1
                test_targets[l, idx] = 1
        loss = loss_fct(logits, test_targets)
        return (loss, outputs) if return_outputs else loss

def train(args):
    seed = args['seed']
    save_dir = args['output_dir']
    logging_dir = args['logging_dir']
    MODEL_NAME = args['MODEL_NAME']
    epochs = args['EPOCH']
    optimizer_name = args['optimizer']
    learning_rate = args['learning_rate']
    batch_size = args['batch_size']
    rtq = args['rtq']
    two_sentence = args['two_sentence']
    except_0 = args['except_0']
    entity_token = args['entity_token']

    seed_everything(seed)

    # load model and tokenizer
    if 'xlm' in MODEL_NAME:
        tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME, additional_special_tokens=['[E1]', '[E2]', '[E1-NER]', '[E2-NER]'])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, additional_special_tokens=['[E1]', '[E2]', '[E1-NER]', '[E2-NER]'])

    if entity_token == 'on':
        train_dataset = ner_load_data("/opt/ml/input/data/train/ner_train_normalize.csv")
        #train_dataset = ner_load_data("/opt/ml/input/data/train/ner_train_ver2.tsv")
    else:
        # load dataset
        train_dataset = load_data("/opt/ml/input/data/train/train.tsv")
    # dev_dataset = load_data("./dataset/train/dev.tsv")
    train_label = train_dataset['label'].values
    # dev_label = dev_dataset['label'].values

    if rtq == 'on':
        tokenized_train = rtq_tokenized_dataset(train_dataset, tokenizer)
        processed_dataset = RtQDataset(tokenized_train, train_label)
    elif two_sentence == 'on':
        tokenized_train = two_sentence_tokenized_dataset(train_dataset, tokenizer)
        processed_dataset = TwoSentenceDataset(tokenized_train, train_label)
    elif entity_token == 'on':
        #tokenized_train = tokenized_dataset(train_dataset, tokenizer)
        #processed_dataset = RE_Dataset(tokenized_train, train_label)
        #tokenized_train = single_tokenized_dataset(train_dataset, tokenizer)
        tokenized_train = ner_tokenized_dataset(train_dataset, tokenizer)
        processed_dataset = RE_Dataset(tokenized_train, train_label)
    else:
        # tokenizing dataset
        tokenized_train = tokenized_dataset(train_dataset, tokenizer)
        # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

        # make dataset for pytorch.
        processed_dataset = RE_Dataset(tokenized_train, train_label)
        # processed_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # setting model hyperparameter
    config = AutoConfig.from_pretrained(MODEL_NAME)
    if 'xlm' in MODEL_NAME:
        config = XLMRobertaConfig.from_pretrained((MODEL_NAME))

    if rtq == 'on':
        config.num_labels = 2
    elif except_0 == 'on':
        config.num_labels = 41
    else:
        config.num_labels = 42
    if 'xlm' in MODEL_NAME:
        model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    model.resize_token_embeddings(len(tokenizer))
    """
    if 'electra' in MODEL_NAME:
        electra_config = ElectraConfig.from_pretrained(MODEL_NAME)
        electra_config.num_label = 42
        model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME, config=electra_config)
    elif 'bert' in MODEL_NAME:
        bert_config = BertConfig.from_pretrained(MODEL_NAME)
        bert_config.num_label = 42
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=bert_config)
    """
    print(processed_dataset[0])
    print(processed_dataset[1])
    model.parameters
    model.to(device)
    #tb_writer = SummaryWriter(log_dir=save_dir)
    #logger = TensorBoardCallback(tb_writer)
    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir=save_dir,  # output directory
        save_total_limit=3,# number of total save model.
        save_strategy='epoch',
        #save_steps=500,  # model saving step.
        num_train_epochs=epochs,  # total number of training epochs
        learning_rate=learning_rate,  # learning_rate
        per_device_train_batch_size=batch_size,  # batch size per device during training
        # per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=logging_dir,  # directory for storing logs
        logging_steps=100,  # log saving step.
        # evaluation_strategy='steps', # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        # eval_steps = 500,            # evaluation step.
        label_smoothing_factor=0.5
    )
    if except_0 == 'on':
        trainer = testTrainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=processed_dataset,  # training dataset
            # eval_dataset=RE_dev_dataset,             # evaluation dataset
            # compute_metrics=compute_metrics         # define metrics function
        )
    else:
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=processed_dataset,  # training dataset
            # eval_dataset=RE_dev_dataset,             # evaluation dataset
            # compute_metrics=compute_metrics         # define metrics function
        )
    # train model
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    args = parser.parse_args()
    with open(args.config_file) as f:
        args = AttrDict(json.load(f))
    print(args)
    train(args)
