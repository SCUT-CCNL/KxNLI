# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from model import Attention_NMT
from data import snli_Data
import numpy as np
from tqdm import tqdm
import config as argumentparser
 
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
config = argumentparser.ArgumentParser()
torch.manual_seed(config.seed)
from nltk.translate.bleu_score import corpus_bleu
if config.cuda and torch.cuda.is_available():
    torch.cuda.set_device(config.gpu)
def get_dev_loss(data_iter):
    model.eval()
    process_bar = tqdm(data_iter)
    loss = 0
    for source_data, target_data_input, target_data,rationales_data,graph_embs,_ in process_bar:
        if config.cuda and torch.cuda.is_available():
            source_data = source_data.cuda()
            rationales_data = rationales_data.cuda()
            target_data_input = target_data_input.cuda()
            target_data = target_data.cuda()
            graph_embs = graph_embs.cuda()
        else:
            source_data = torch.autograd.Variable(source_data).long()
            rationales_data = torch.autograd.Variable(rationales_data).long()
            target_data_input = torch.autograd.Variable(target_data_input).long()
        target_data = torch.autograd.Variable(target_data).squeeze()
        out = model(source_data, target_data_input, rationales_data, graph_embs)
        loss_now = criterion(out.view(-1, config.target_vocab_size), autograd.Variable(target_data.view(-1).long()))
        weights = target_data.view(-1) != 0
        loss_now = torch.sum((loss_now * weights.float())) / torch.sum(weights.float())
        loss+=loss_now.data.item()
    return loss

def get_test_bleu(data_iter, type='test', model_type='std'):
    model.eval()
    process_bar = tqdm(data_iter)
    refs = []
    sour = []
    preds = []
    label = []
    for source_data, target_data_input, target_data, rationales_data,graph_embs, labels in process_bar:
        target_input = torch.Tensor(np.zeros([source_data.shape[0], 1])+2)
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
                if out[i][j]!=3:
                    tmp_preds[i].append(out[i][j])
                else:
                    break
        preds += tmp_preds

        tmp_refs = []
        for i in range(target_data.shape[0]):
            tmp_refs.append([])
        for i in range(target_data.shape[0]):
            for j in range(target_data.shape[1]):
                if target_data[i][j]!=3 and target_data[i][j]!=0:
                    tmp_refs[i].append(target_data[i][j])
        tmp_refs = [[x] for x in tmp_refs]
        refs+=tmp_refs

        sour_data_for_show = []
        for i in range(source_data.shape[0]):
            sour_data_for_show.append([])
        for i in range(source_data.shape[0]):
            for j in range(source_data.shape[1]):
                if source_data[i][j]!=3 and source_data[i][j]!=0:
                    sour_data_for_show[i].append(source_data[i][j].item())
        sour_data_for_show = [[x] for x in sour_data_for_show]
        sour+=sour_data_for_show

        labels_for_show = []
        for i in range(source_data.shape[0]):
            labels_for_show.append([])
        labels_for_show = [[x] for x in labels]
        label+=labels_for_show

    bleu = corpus_bleu(refs,preds)*100

    with open(type+".txt","w") as f:
        for i in range(len(preds)):
            tmp_ref = [target_id2word[id] for id in refs[i][0]]
            tmp_pred = [target_id2word[id] for id in preds[i]]
            sd = [source_id2word[id] for id in sour[i][0]]
            sd.reverse()
            f.write("Golden_explanation: "+" ".join(tmp_ref)+"\n")
            f.write("Pred_explanation: "+" ".join(tmp_pred)+"\n")
            f.write("Premise_hypothesis: "+" ".join(sd)+"\n")
            f.write("label: "+label[i][0]+"\n")
            f.write("\n\n")
    return bleu

# training_set = snli_Data()
training_set = snli_Data(source_data_name="train.csv",target_data_name="train.csv",
                          source_vocab_size=config.source_vocab_size, target_vocab_size=config.target_vocab_size)
training_iter = torch.utils.data.DataLoader(dataset=training_set,
                                            batch_size=config.batch_size,
                                            shuffle=True,
                                            num_workers=0)
valid_set = snli_Data(source_data_name="dev.csv",target_data_name="dev.csv",
                       source_vocab_size=config.source_vocab_size, target_vocab_size=config.target_vocab_size)
valid_iter = torch.utils.data.DataLoader(dataset=valid_set,
                                            batch_size=config.batch_size,
                                            shuffle=True,
                                            num_workers=0)
test_set = snli_Data(source_data_name="test.csv",target_data_name="test.csv",
                      source_vocab_size=config.source_vocab_size, target_vocab_size=config.target_vocab_size)
test_iter = torch.utils.data.DataLoader(dataset=test_set,
                                            batch_size=config.batch_size,
                                            shuffle=True,
                                            num_workers=0)
model = Attention_NMT(source_vocab_size=config.source_vocab_size,target_vocab_size=config.target_vocab_size,embedding_size=config.embedding_size,
                 source_length=config.source_length,target_length=config.target_length,lstm_size=config.lstm_size, graph_size=config.graph_emb_size)
if config.cuda and torch.cuda.is_available():
    model.cuda()
criterion = nn.CrossEntropyLoss(reduce=False)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
loss = -1

source_id2word = dict([[x[1],x[0]] for x in valid_set.source_word2id.items()])
target_id2word = dict([[x[1],x[0]] for x in valid_set.target_word2id.items()])
best_epoch = 0
best_test_bleu = 0
for epoch in range(config.epoch):
    model.train()
    process_bar = tqdm(training_iter, desc='epoch '+str(epoch))
    for source_data, target_data_input, target_data, rationales_data, graph_embs, _ in process_bar:
        if config.cuda and torch.cuda.is_available():
            source_data = source_data.cuda()
            target_data_input = target_data_input.cuda()
            rationales_data = rationales_data.cuda()
            target_data = target_data.cuda()
            # graph_embs = graph_embs.cuda()
        else:
            source_data = torch.autograd.Variable(source_data).long()
            rationales_data = torch.autograd.Variable(rationales_data).long()
            target_data_input = torch.autograd.Variable(target_data_input).long()
            graph_embs = torch.autograd.Variable(graph_embs).long()
        target_data = torch.autograd.Variable(target_data).squeeze()
        out = model(source_data,target_data_input,rationales_data, graph_embs)

        loss_now = criterion(out.view(-1,config.target_vocab_size), autograd.Variable(target_data.view(-1).long()))
        weights = target_data.view(-1)!=0
        loss_now = torch.sum((loss_now*weights.float()))/torch.sum(weights.float())
        if loss == -1:
            loss = loss_now.data.item()
        else:
            loss = 0.95*loss+0.05*loss_now.data.item()
        process_bar.set_postfix(loss=loss_now.data.item())
        process_bar.update()
        optimizer.zero_grad()
        loss_now.backward()
        optimizer.step()

    model_type = 'std'
    data_type = 'test'
    test_bleu = get_test_bleu(test_iter, data_type, model_type)
    if test_bleu>best_test_bleu:
        torch.save(model, 'model_'+model_type+'.pk')
        print("test bleu is:", test_bleu)
        best_test_bleu = test_bleu
        best_epoch = epoch
    if (epoch+1) % 10 == 0:
        torch.save(model, './saved_models/model_' + model_type + str(epoch) + '.pk')
        print('saved model...', str(epoch), '.pk')
    print('the best epoch is ',str(best_epoch))
    valid_loss = get_dev_loss(valid_iter)
    print ("valid loss is:",valid_loss)
