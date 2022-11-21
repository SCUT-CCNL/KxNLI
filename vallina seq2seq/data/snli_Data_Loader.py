#coding:utf-8
from torch.utils import data
import os
import nltk
import numpy as np
import pickle
from collections import Counter
import csv
import re
import json
import tqdm

graph=True
global concept2id, cpts2emb
concept2id = {}
cpts2emb = []
if graph:
    print('loading embed_cpt.vec')
    with open("./data/esnli/embed_cpt.vec", "r", encoding="utf8") as f_cpts_emb:
        dic = json.load(f_cpts_emb)
        cpts2emb = dic['ent_embeddings.weight']
        rel2emb = dic['rel_embeddings.weight']

    print('loading the concept2id')
    with open('./data/esnli/concept.txt', "r", encoding="utf8") as f:
        for w in f.readlines():
            concept2id[w.strip()] = len(concept2id)
    print('done')

class snli_Data(data.DataLoader):
    def __init__(self,source_data_name="train.csv",target_data_name="train.csv",
                 source_vocab_size = 60000, target_vocab_size = 60000, use_graph=True):
        self.use_graph = use_graph
        self.source_data_name = source_data_name
        self.target_data_name = target_data_name
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.source_data, self.target_data_input, self.target_data, \
        self.rationales_data, self.graph_embs, self.labels = self.load_data(with_rationale=True)

    def load_data(self, with_rationale=True):
        global cpts2emb, concept2id

        raw_source_data, raw_target_data, raw_rationales, graph_embs, labels = [], [], [], [], []
        with open("./data/esnli/raw_data/"+self.source_data_name, 'r') as csv_file, \
                open("./data/esnli/cpts/golden/1hops_triple_"+self.source_data_name.replace('.csv','.json'), 'r') as f_cpts:
            csv_reader = csv.reader(csv_file, delimiter=',')
            cpt_lines = f_cpts.readlines()
            for row_count, row in enumerate(csv_reader):
                if row_count == 0:
                    continue
                cpts = json.loads(cpt_lines[row_count - 1])['concepts']
                if cpts != []:
                    cpt2ids = [concept2id[c] for c in cpts]
                    graph_emb = np.mean([cpts2emb[c] for c in cpt2ids], axis=0)
                    graph_embs.append(graph_emb)
                    raw_source_data.append(row[2] + ' ' + row[3])
                    raw_target_data.append(row[4])
                    labels.append(row[1])
                    if with_rationale:
                        if 'train' in self.source_data_name:
                            raw_rationales.append((' '.join(re.findall(r'\*(.*?)\*', row[6]+' '+row[7]))))
                        else:
                            raw_rationales.append(( ' '.join(re.findall(r'\*(.*?)\*', row[5]+' '+row[6]))))
                else: continue

        assert len(raw_target_data)==len(raw_source_data)==len(graph_embs)==len(labels)
        print (len(raw_target_data))
        source_data, target_data, rationales_data = [], [], []
        for i in range(len(raw_source_data)):
            source_sentence = nltk.word_tokenize(raw_source_data[i],language="english")
            target_sentence = nltk.word_tokenize(raw_target_data[i],language="english")
            rationales_sent = nltk.word_tokenize(raw_rationales[i],language="english")
            if len(source_sentence)<=100 and len(target_sentence)<=100:
                source_data.append(source_sentence)
                target_data.append(target_sentence)
                rationales_data.append(rationales_sent)
        if not os.path.exists( "./data/esnli/source_word2id"):
            source_word2id = self.get_word2id(source_data,self.source_vocab_size)
            target_word2id = self.get_word2id(target_data,self.target_vocab_size)
            rational_word2id = source_word2id
            self.source_word2id = source_word2id
            self.target_word2id = target_word2id
            pickle.dump(source_word2id, open("./data/esnli/source_word2id", "wb"))
            pickle.dump(target_word2id, open("./data/esnli/target_word2id", "wb"))
        else:
            self.source_word2id = pickle.load(open("./data/esnli/word2id", "rb"))
            self.target_word2id = pickle.load(open("./data/esnli//word2id", "rb"))
        source_data = self.get_id_datas(source_data,self.source_word2id)
        rationales_data = self.get_id_datas(rationales_data,self.source_word2id,is_source=False)
        target_data = self.get_id_datas(target_data,self.target_word2id,is_source=False)

        target_data_input = [[2]+sentence[0:-1] for sentence in target_data]
        source_data = np.array(source_data)
        rationales_data = np.array(rationales_data)
        target_data = np.array(target_data)
        target_data_input = np.array(target_data_input)

        return source_data,target_data_input,target_data,rationales_data,graph_embs, labels
    def get_word2id(self,data,word_num):
        words = []
        for sentence in data:
            for word in sentence:
                words.append(word)
        word_freq = dict(Counter(words).most_common(word_num-4))
        word2id = {"<pad>":0,"<unk>":1,"<start>":2,"<end>":3}
        for word in word_freq:
            word2id[word] = len(word2id)
        return word2id
    def get_id_datas(self,datas,word2id,is_source = True):
        for i, sentence in enumerate(datas):
            for j, word in enumerate(sentence):
                datas[i][j] = word2id.get(word,1)
            if is_source:
                datas[i] = datas[i][0:64] +[0]*(64-len(datas[i]))
                datas[i].reverse()
            else:
                datas[i] = datas[i][0:63]+ [3] + [0] * (63 - len(datas[i]))
        return datas


    def __getitem__(self, idx):
        if graph:
            return self.source_data[idx], self.target_data_input[idx], self.target_data[idx],\
               self.rationales_data[idx], self.graph_embs[idx], self.labels[idx]
        else:
            return self.source_data[idx], self.target_data_input[idx], self.target_data[idx],\
               self.rationales_data[idx], self.labels[idx]

    def __len__(self):
        return len(self.source_data)

class snli_Data_noR(data.DataLoader):
    def __init__(self,source_data_name="train.csv",target_data_name="train.csv",
                 source_vocab_size = 60000, target_vocab_size = 60000, use_graph=True):
        self.use_graph = use_graph
        self.source_data_name = source_data_name
        self.target_data_name = target_data_name
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.source_data, self.target_data_input, self.target_data, \
        self.graph_embs, self.labels = self.load_data()

    def load_data(self):
        global cpts2emb, concept2id
        raw_source_data, raw_target_data, graph_embs, labels = [], [], [], []
        with open("./data/esnli/raw_data/"+self.source_data_name, 'r') as csv_file, \
                open("./data/esnli/cpts/golden/1hops_triple_"+self.source_data_name.replace('.csv','.json'), 'r') as f_cpts:
            csv_reader = csv.reader(csv_file, delimiter=',')
            cpt_lines = f_cpts.readlines()
            for row_count, row in enumerate(csv_reader):
                if row_count == 0:
                    continue
                cpts = json.loads(cpt_lines[row_count-1])['concepts']
                if cpts != []:
                    cpt2ids = [concept2id[c] for c in cpts]
                    graph_emb = np.mean([cpts2emb[c] for c in cpt2ids], axis=0)
                    graph_embs.append(graph_emb)
                    raw_source_data.append(row[2] + ' ' + row[3])
                    raw_target_data.append(row[4])
                    labels.append(row[1])

        assert len(raw_target_data)==len(raw_source_data)==len(graph_embs)==len(labels)
        print (len(raw_target_data))
        source_data, target_data = [], []
        for i in range(len(raw_source_data)):
            source_sentence = nltk.word_tokenize(raw_source_data[i],language="english")
            target_sentence = nltk.word_tokenize(raw_target_data[i],language="english")
            source_data.append(source_sentence)
            target_data.append(target_sentence)
        if not os.path.exists( "./data/esnli/source_word2id"):
            source_word2id = self.get_word2id(source_data,self.source_vocab_size)
            target_word2id = self.get_word2id(target_data,self.target_vocab_size)
            self.source_word2id = source_word2id
            self.target_word2id = target_word2id
            pickle.dump(source_word2id, open("./data/esnli/source_word2id", "wb"))
            pickle.dump(target_word2id, open("./data/esnli/target_word2id", "wb"))
        else:
            self.source_word2id = pickle.load(open("./data/esnli/word2id", "rb"))
            self.target_word2id = pickle.load(open("./data/esnli//word2id", "rb"))
        source_data = self.get_id_datas(source_data,self.source_word2id)
        target_data = self.get_id_datas(target_data,self.target_word2id,is_source=False)

        target_data_input = [[2]+sentence[0:-1] for sentence in target_data]
        source_data = np.array(source_data)
        target_data = np.array(target_data)
        target_data_input = np.array(target_data_input)

        return source_data,target_data_input,target_data,graph_embs, labels
    def get_word2id(self,data,word_num):
        words = []
        for sentence in data:
            for word in sentence:
                words.append(word)
        word_freq = dict(Counter(words).most_common(word_num-4))
        word2id = {"<pad>":0,"<unk>":1,"<start>":2,"<end>":3}
        for word in word_freq:
            word2id[word] = len(word2id)
        return word2id
    def get_id_datas(self,datas,word2id,is_source = True):
        for i, sentence in enumerate(datas):
            for j, word in enumerate(sentence):
                datas[i][j] = word2id.get(word,1)
            if is_source:
                datas[i] = datas[i][0:64] +[0]*(64-len(datas[i]))
                datas[i].reverse()
            else:
                datas[i] = datas[i][0:63]+ [3] + [0] * (63 - len(datas[i]))
        return datas



    def __getitem__(self, idx):
        return self.source_data[idx], self.target_data_input[idx], self.target_data[idx],\
               self.graph_embs[idx], self.labels[idx]

    def __len__(self):
        return len(self.source_data)