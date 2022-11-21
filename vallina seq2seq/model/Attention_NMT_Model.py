# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import json

class Attention_NMT(nn.Module):

    def __init__(self,source_vocab_size,target_vocab_size,lstm_size,graph_size,batch_size = 32):
        super(Attention_NMT,self).__init__()
        self.batch_size = batch_size
        self.source_embedding =nn.Embedding(source_vocab_size,embedding_size)
        self.target_embedding = nn.Embedding(target_vocab_size,embedding_size)
        self.encoder = nn.LSTM(input_size=embedding_size,hidden_size=lstm_size,num_layers=1,
                               bidirectional=True,batch_first=True)
        self.decoder = nn.LSTM(input_size=embedding_size+4*lstm_size,hidden_size=lstm_size,num_layers=1,
                               batch_first=True)
        self.attention_fc_1 = nn.Linear(3*lstm_size, 3*lstm_size)
        self.attention_fc_2 = nn.Linear(3 * lstm_size, 1)
        self.class_fc_1 = nn.Linear(embedding_size+4*lstm_size+lstm_size, 2*lstm_size)
        self.class_fc_graph = nn.Linear(graph_size+embedding_size+4*lstm_size+lstm_size, 2*lstm_size)
        self.class_fc_2 = nn.Linear(2*lstm_size, target_vocab_size)

    def attention_forward(self,input_embedding,dec_prev_hidden,enc_output,enc_output_r):
        prev_dec_h = dec_prev_hidden[0].squeeze().unsqueeze(1).repeat(1, 64, 1)

        atten_input = torch.cat([enc_output, prev_dec_h], dim=-1)
        attention_weights = self.attention_fc_2(F.relu(self.attention_fc_1(atten_input)))
        attention_weights = F.softmax(attention_weights, dim=1)
        atten_output = torch.sum(attention_weights * enc_output, dim=1).unsqueeze(1)

        atten_input_r = torch.cat([enc_output_r, prev_dec_h], dim=-1)
        attention_weights_r = self.attention_fc_2(F.relu(self.attention_fc_1(atten_input_r)))
        attention_weights_r = F.softmax(attention_weights_r, dim=1)
        atten_output_r = torch.sum(attention_weights_r * enc_output_r, dim=1).unsqueeze(1)

        dec_lstm_input = torch.cat([input_embedding, atten_output, atten_output_r], dim=2)
        dec_output, dec_hidden = self.decoder(dec_lstm_input, dec_prev_hidden)
        return atten_output,atten_output_r,dec_output,dec_hidden
    def forward(self, source_data,target_data,rationales, graph_embs, mode = "train",is_gpu=True):
        global cpts2emb
        source_data_embedding = self.source_embedding(source_data)

        rationales_embedding = self.source_embedding(rationales)
        try:
            graph_embs = graph_embs.to(torch.float32)
        except:
            graph_embs = torch.stack(graph_embs, dim=1).to(torch.float32)

        enc_output, enc_hidden = self.encoder(source_data_embedding)
        enc_output_r, enc_hidden_r = self.encoder(rationales_embedding)
        self.atten_outputs = Variable(torch.zeros(target_data.shape[0],
                                                  target_data.shape[1],
                                                  enc_output.shape[2]))
        self.atten_outputs_r = Variable(torch.zeros(target_data.shape[0],
                                                  target_data.shape[1],
                                                  enc_output.shape[2]))
        self.dec_outputs = Variable(torch.zeros(target_data.shape[0],
                                                target_data.shape[1],
                                                enc_hidden[0].shape[2]))
        if is_gpu:
            self.atten_outputs = self.atten_outputs.cuda()
            self.atten_outputs_r = self.atten_outputs_r.cuda()
            self.dec_outputs = self.dec_outputs.cuda()
            graph_embs = graph_embs.unsqueeze(1).cuda()

        if mode=="train":
            graph_embs = graph_embs.repeat(1, 64, 1)
            target_data_embedding = self.target_embedding(target_data)
            dec_prev_hidden = [enc_hidden[0][0].unsqueeze(0),enc_hidden[1][0].unsqueeze(0)]


            for i in range(32):
                input_embedding = target_data_embedding[:,i,:].unsqueeze(1)
                atten_output,atten_output_r, dec_output, dec_hidden = self.attention_forward(input_embedding,
                                                                              dec_prev_hidden,
                                                                              enc_output,
                                                                              enc_output_r)
                self.atten_outputs[:,i] = atten_output.squeeze()
                self.atten_outputs_r[:,i] = atten_output_r.squeeze()
                self.dec_outputs[:,i] = dec_output.squeeze()
                dec_prev_hidden = dec_hidden

            class_input = torch.cat([target_data_embedding, self.atten_outputs, self.atten_outputs_r,
                                     graph_embs, self.dec_outputs], dim=2)
            outs = self.class_fc_2(F.relu(self.class_fc_graph(class_input)))

        else:
            input_embedding = self.target_embedding(target_data)
            dec_prev_hidden = [enc_hidden[0][0].unsqueeze(0),enc_hidden[1][0].unsqueeze(0)]
            outs = []
            for i in range(32):
                atten_output, atten_output_r, dec_output, dec_hidden = self.attention_forward(input_embedding,
                                                                              dec_prev_hidden,
                                                                              enc_output,
                                                                              enc_output_r)
                class_input = torch.cat([input_embedding, atten_output, atten_output_r,
                                         graph_embs, dec_output], dim=2)
                pred = self.class_fc_2(F.relu(self.class_fc_graph(class_input)))

                pred = torch.argmax(pred,dim=-1)
                outs.append(pred.squeeze().cpu().numpy())
                dec_prev_hidden = dec_hidden
                input_embedding = self.target_embedding(pred)
        return outs
