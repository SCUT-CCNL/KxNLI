import re
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

import json
import random
import pickle
import logging
import csv

from sklearn.metrics import classification_report

from transformers import *
from model_classify import BaseSNLI_roberta

from tqdm import tqdm
from transformers import RobertaTokenizer
# BaseSNLlogging.getLogger().setLevel(print)


# device =''
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.set_device(1)

import argparse
model_to_save = 'selector_bert_withHints_3'

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', default='../data/snli/train/source.csv', type=str)
parser.add_argument('--dev_data', default='../data/snli/dev/source.csv', type=str)
parser.add_argument('--test_data', default='../data/snli/test/source.csv', type=str)

parser.add_argument('--model_to_save',default=model_to_save, type=str)
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--n_epoch', default=20, type=int)
args = parser.parse_args()

def load_dataset(ph_path, expl_path):
    data = []
    count = 0
    expl_f = open(expl_path,'r')
    expl_set = expl_f.readlines()
    with open(ph_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row_count, row in enumerate(csv_reader):
            count += 1
            if count == 1:
                continue

            gloden_label = row[1]
            premise = row[2]
            hypothesis = row[3]
            # expl = expl_set[row_count-1].replace('\n','')
            data.append(premise+'\t'+hypothesis+'\t'+expl+'\t'+gloden_label)
    return data

def load_all_dataset():
    dev_data = load_dataset(args.dev_data, '../models/snli/att-medium-withR-snli_eval/dev.txt')
    train_data = load_dataset(args.train_data, '../models/snli/att-medium-withR-snli_eval/golden/train.txt')
    test_data = load_dataset(args.test_data, '../models/snli/att-medium-withR-snli_eval/test.txt')
    return train_data, dev_data, test_data

def packing(d):
    max_length = max([len(item) for item in d['input_ids']])
    for i in range(len(d['input_ids'])):
        diff = max_length - len(d['input_ids'][i])
        for _ in range(diff):
            d['input_ids'][i].append(1)
            d['attention_mask'][i].append(0)
    return d

print('bert-golden-phe')
def prepare_batch(batch):
    lbs = [label2idx[d.split('\t')[3]] for d in batch]
    d_input = {'input_ids':[], 'attention_mask':[]}
    for i in range(len(batch)):
        b = batch[i].split('\t')
        text = b[2]
        d_cur = tokenizer(text)
        d_input['input_ids'].append(d_cur['input_ids'])
        d_input['attention_mask'].append(d_cur['attention_mask'])
    d_input = packing(d_input)
    return d_input, lbs


def train(batch):
    optimizer.zero_grad()
    d, lbs = prepare_batch(batch)
    logits = model(d)
    loss = F.cross_entropy(logits, torch.LongTensor(lbs).to(device))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    return loss.item()


def evaluate(data):
    gold, pred = [], []
    selector_pred = []
    with torch.no_grad():
        batches = [data[x:x + batch_size] for x in range(0, len(data), batch_size)]
        for batch_no, batch in enumerate(batches):
            d, lbs = prepare_batch(batch)
            logits = model(d)

            _, idx = torch.max(logits, 1)
            gold.extend(lbs)
            pred.extend(idx.tolist())

    print(classification_report(
        gold, pred, target_names=list(label2idx.keys()), digits=4
    ))

    report = classification_report(
        gold, pred, target_names=list(label2idx.keys()), output_dict=True, digits=4
    )
    return report['accuracy']

if __name__=='__main__':
    label2idx = {'entailment':0, 'neutral':1, 'contradiction':2}
    idx2label = {v:k for k,v in label2idx.items()}
    train_data, dev_data, test_data = load_all_dataset()

    batch_size = args.batch_size
    lr = args.lr
    n_epoch = args.n_epoch

    model_name = 'roberta-base'

    config = RobertaConfig.from_pretrained(model_name)
    config.num_labels = 3

    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    model = BaseSNLI_roberta(config).to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    num_batches = len([train_data[x:x + batch_size] for x in range(0, len(train_data), batch_size)])

    num_training_steps = n_epoch * num_batches

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=150, num_training_steps=num_training_steps
    )

    best_f1_dev, best_epoch_dev = 0, 0

    for epoch in range(n_epoch):
        prev_lr = lr

        random.shuffle(train_data)
        batches = [train_data[x:x + batch_size] for x in range(0, len(train_data), batch_size)]
        process_bar = tqdm(batches, desc='epoch:' + str(epoch))
        model.train()
        current_loss, seen_sentences, modulo = 0.0, 0, max(1, int(len(batches) / 10))
        for batch_no, sent_batch in enumerate(process_bar):
            batch_loss = train(sent_batch)
            current_loss += batch_loss
            seen_sentences += len(sent_batch)
        current_loss /= len(train_data)
        process_bar.set_postfix(loss=current_loss)
        process_bar.update()

        model.eval()
        print('-' * 100)
        print('---------- dev data ---------')
        f1_dev = evaluate(dev_data)
        if f1_dev > best_f1_dev:
            best_f1_dev = f1_dev
            best_epoch_dev = epoch
        print('best acc: {}, best epoch: {}'.format(best_f1_dev, best_epoch_dev))

        print('---------- test data ---------')
        f1 = evaluate(test_data)


