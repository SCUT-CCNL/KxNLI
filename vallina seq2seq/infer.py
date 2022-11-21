# -*- coding: utf-8 -*-
import torch
import json
import torch.nn as nn
import torch.optim as optim
from model import Attention_NMT
from data import snli_Data
import numpy as np
from tqdm import tqdm
import config as argumentparser

config = argumentparser.ArgumentParser()
torch.manual_seed(config.seed)
from nltk.translate.bleu_score import corpus_bleu
if config.cuda and torch.cuda.is_available():
    torch.cuda.set_device(config.gpu)

def get_test_bleu(data_iter, type='dev'):
    model = torch.load('./saved_model_std.pk')
    model.cuda()
    model.eval()
    refs, sour, preds, label, prem_with_r, hypo_with_r = [], [], [], [], [], []

    process_bar = tqdm(data_iter)
    for source_data, target_data_input, target_data, rationales_data, \
        graph_embs, labels, prem_with_rs, hypo_with_rs in process_bar:
        target_input = torch.Tensor(np.zeros([source_data.shape[0], 1]) + 2)
        if config.cuda and torch.cuda.is_available():
            source_data = source_data.cuda()
            target_input = target_input.cuda().long()
            rationales_data = rationales_data.cuda().long()
            graph_embs = graph_embs.cuda()
        else:
            source_data = torch.autograd.Variable(source_data).long()
            target_input = torch.autograd.Variable(target_input).long()
            rationales_data = torch.autograd.Variable(rationales_data).long()
        target_data = target_data.numpy()
        out = model(source_data, target_input, rationales_data, graph_embs, mode="test")
        out = np.array(out).T

        tmp_preds = []
        for i in range(out.shape[0]):
            tmp_preds.append([])
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if out[i][j] != 3:
                    tmp_preds[i].append(out[i][j])
                else:
                    break
        preds += tmp_preds

        tmp_refs = []
        for i in range(target_data.shape[0]):
            tmp_refs.append([])
        for i in range(target_data.shape[0]):
            for j in range(target_data.shape[1]):
                if target_data[i][j] != 3 and target_data[i][j] != 0:
                    tmp_refs[i].append(target_data[i][j])
        tmp_refs = [[x] for x in tmp_refs]
        refs += tmp_refs

        labels_for_show = []
        for i in range(source_data.shape[0]):
            labels_for_show.append([])
        labels_for_show = [[x] for x in labels]
        label += labels_for_show

        prem_for_show = []
        for i in range(source_data.shape[0]):
            prem_for_show.append([])
        prem_for_show = [[x] for x in prem_with_rs]
        prem_with_r += prem_for_show

        hypo_for_show = []
        for i in range(source_data.shape[0]):
            hypo_for_show.append([])
        hypo_for_show = [[x] for x in hypo_with_rs]
        hypo_with_r += hypo_for_show

        sour_data_for_show = []
        for i in range(source_data.shape[0]):
            sour_data_for_show.append([])
        for i in range(source_data.shape[0]):
            for j in range(source_data.shape[1]):
                if source_data[i][j] != 3 and source_data[i][j] != 0:
                    sour_data_for_show[i].append(source_data[i][j].item())
        sour_data_for_show = [[x] for x in sour_data_for_show]
        sour += sour_data_for_show

    bleu = corpus_bleu(refs, preds) * 100

    with open(type + ".txt", "w") as f:
        for i in range(len(preds)):
            d = {}
            tmp_pred = [target_id2word[id] for id in preds[i]]
            d['premise'] = prem_with_r[i][0]
            d['hypothesis'] = hypo_with_r[i][0]
            d['gold_label'] = label[i][0]
            d['expl'] = " ".join(tmp_pred)
            f.write(json.dumps(d) + '\n')

    return bleu


# training_set = snli_Data()
training_set = snli_Data(source_data_name="train.csv",target_data_name="train.csv",
                          source_vocab_size=config.source_vocab_size, target_vocab_size=config.target_vocab_size)
training_iter = torch.utils.data.DataLoader(dataset=training_set,
                                            batch_size=config.batch_size,
                                            shuffle=False,
                                            num_workers=0)
valid_set = snli_Data(source_data_name="dev.csv",target_data_name="dev.csv",
                       source_vocab_size=config.source_vocab_size, target_vocab_size=config.target_vocab_size)
valid_iter = torch.utils.data.DataLoader(dataset=valid_set,
                                            batch_size=config.batch_size,
                                            shuffle=False,
                                            num_workers=0)
test_set = snli_Data(source_data_name="test.csv",target_data_name="test.csv",
                      source_vocab_size=config.source_vocab_size, target_vocab_size=config.target_vocab_size)
test_iter = torch.utils.data.DataLoader(dataset=test_set,
                                            batch_size=config.batch_size,
                                            shuffle=False,
                                            num_workers=0)

source_id2word = dict([[x[1],x[0]] for x in valid_set.source_word2id.items()])
target_id2word = dict([[x[1],x[0]] for x in valid_set.target_word2id.items()])
print(len(target_id2word))
print(len(source_id2word))
get_test_bleu(test_iter,'test7-debug')
get_test_bleu(valid_iter,'dev7-debug')
get_test_bleu(training_iter,'train7-debug')