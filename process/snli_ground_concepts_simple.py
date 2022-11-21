import configparser
import json
import csv
import spacy
from spacy.matcher import Matcher
import sys
import timeit
from tqdm import tqdm
import numpy as np
import multiprocessing
import sys

blacklist = set(
    ["from", "as", "more", "either", "in", "and", "on", "an", "when", "too", "to", "i", "do", "can", "be", "that", "or",
     "the", "a", "of", "for", "is", "was", "the", "-PRON-", "actually", "likely", "possibly", "want",
     "make", "my", "someone", "sometimes_people", "sometimes", "would", "want_to",
     "one", "something", "sometimes", "everybody", "somebody", "could", "could_be", "mine", "us", "em",
     "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "A", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst",
     "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af",
     "affected", "affecting", "after", "afterwards", "ag", "again", "against", "ah", "ain", "aj", "al", "all", "allow",
     "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst",
     "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone",
     "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appreciate", "approximately", "ar", "are",
     "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "au", "auth", "av", "available", "aw",
     "away", "awfully", "ax", "ay", "az", "b", "B", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "been",
     "before", "beforehand", "beginnings", "behind", "below", "beside", "besides", "best", "between", "beyond", "bi",
     "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but",
     "bx", "by", "c", "C", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "cc", "cd", "ce", "certain",
     "certainly", "cf", "cg", "ch", "ci", "cit", "cj", "cl", "clearly", "cm", "cn", "co", "com", "come", "comes", "con",
     "concerning", "consequently", "consider", "considering", "could", "couldn", "couldnt", "course", "cp", "cq", "cr",
     "cry", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d", "D", "d2", "da", "date", "dc", "dd", "de", "definitely",
     "describe", "described", "despite", "detail", "df", "di", "did", "didn", "dj", "dk", "dl", "do", "does", "doesn",
     "doing", "don", "done", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "E",
     "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "eg", "ei", "eight", "eighty", "either", "ej", "el",
     "eleven", "else", "elsewhere", "em", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es",
     "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone",
     "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "F", "f2", "fa", "far", "fc", "few",
     "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "five", "fix", "fj", "fl", "fn", "fo", "followed",
     "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front",
     "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "G", "ga", "gave", "ge", "get", "gets", "getting",
     "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr",
     "greetings", "gs", "gy", "h", "H", "h2", "h3", "had", "hadn", "happens", "hardly", "has", "hasn", "hasnt", "have",
     "haven", "having", "he", "hed", "hello", "help", "hence", "here", "hereafter", "hereby", "herein", "heres",
     "hereupon", "hes", "hh", "hi", "hid", "hither", "hj", "ho", "hopefully", "how", "howbeit", "however", "hr", "hs",
     "http", "hu", "hundred", "hy", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "ie", "if",
     "ig", "ignored", "ih", "ii", "ij", "il", "im", "immediately", "in", "inasmuch", "inc", "indeed", "index",
     "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "inward",
     "io", "ip", "iq", "ir", "is", "isn", "it", "itd", "its", "iv", "ix", "iy", "iz", "j", "J", "jj", "jr", "js", "jt",
     "ju", "just", "k", "K", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "ko", "l", "L", "l2", "la", "largely",
     "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets",
     "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr",
     "ls", "lt", "ltd", "m", "M", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me",
     "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mill", "million", "mine", "miss", "ml", "mn", "mo",
     "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "my",
     "n", "N", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "neither",
     "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none",
     "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "novel", "now", "nowhere", "nr", "ns", "nt",
     "ny", "o", "O", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi",
     "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq",
     "or", "ord", "os", "ot", "otherwise", "ou", "ought", "our", "out", "outside", "over", "overall", "ow", "owing",
     "own", "ox", "oz", "p", "P", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular",
     "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed",
     "please", "plus", "pm", "pn", "po", "poorly", "pp", "pq", "pr", "predominantly", "presumably", "previously",
     "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "Q", "qj", "qu",
     "que", "quickly", "quite", "qv", "r", "R", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really",
     "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively",
     "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm",
     "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "S", "s2", "sa", "said", "saw", "say", "saying",
     "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "seem", "seemed", "seeming", "seems", "seen",
     "sent", "seven", "several", "sf", "shall", "shan", "shed", "shes", "show", "showed", "shown", "showns", "shows",
     "si", "side", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somehow",
     "somethan", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified",
     "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially",
     "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "sz", "t", "T", "t1", "t2", "t3", "take",
     "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx",
     "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter",
     "thereby", "thered", "therefore", "therein", "thereof", "therere", "theres", "thereto", "thereupon", "these",
     "they", "theyd", "theyre", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou",
     "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip",
     "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried",
     "tries", "truly", "try", "trying", "ts", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "U", "u201d",
     "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto",
     "uo", "up", "upon", "ups", "ur", "us", "used", "useful", "usefully", "usefulness", "using", "usually", "ut", "v",
     "V", "va", "various", "vd", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt",
     "vu", "w", "W", "wa", "was", "wasn", "wasnt", "way", "we", "wed", "welcome", "well", "well-b", "went", "were",
     "weren", "werent", "what", "whatever", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas",
     "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who",
     "whod", "whoever", "whole", "whom", "whomever", "whos", "whose", "why", "wi", "widely", "with", "within",
     "without", "wo", "won", "wonder", "wont", "would", "wouldn", "wouldnt", "www", "x", "X", "x1", "x2", "x3", "xf",
     "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "Y", "y2", "yes", "yet", "yj", "yl", "you",
     "youd", "your", "youre", "yours", "yr", "ys", "yt", "z", "Z", "zero", "zi", "zz"])

concept_vocab = set()
config = configparser.ConfigParser()
config.read("paths.cfg")
with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
    cpnet_vocab = [l.strip() for l in list(f.readlines())]
cpnet_vocab = set([c.replace("_", " ") for c in cpnet_vocab])

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])


def hard_ground(sent):
    global cpnet_vocab, model_vocab
    sent = sent.lower()
    doc = nlp(sent)
    res = set()
    for t in doc:
        if t.lemma_ in cpnet_vocab and t.lemma_ not in blacklist \
                and t.lemma_ in model_vocab:
            if t.pos_ == "NOUN" or t.pos_ == "VERB":
                res.add(t.lemma_)
    return res


# def match(input):
#     return match_mentioned_concepts(input[0], input[1])


def match_mentioned_concepts(question, answer, labels):

    res = []
    error_ids = []
    for sid, str_q_star in tqdm(enumerate(question), total=len(question)):  # , desc="grounding batch_id:%d"%batch_id):
        str_a_star = answer[sid]
        label = labels[sid]
        str_q = str_q_star.replace('[ ','').replace(' ]','')
        str_a = str_a_star.replace('[ ','').replace(' ]','')

        all_concepts = hard_ground(str_q + ' ' + str_a)
        q_concepts = hard_ground(str_q)
        a_concepts = all_concepts - q_concepts
        if list(q_concepts) == [] and list(a_concepts)  == []:
            error_ids.append(sid+1)
        else:
            res.append({"sent": str_q_star.replace('[ ','*').replace(' ]','*'),
                        "ans": str_a_star.replace('[ ','*').replace(' ]','*'),
                    "qc": list(q_concepts), "ac": list(a_concepts), "label":label})

    return res, error_ids



def lemmatize(nlp, concept):
    doc = nlp(concept.replace("_", " "))
    lcs = set()

    lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
    return lcs


def load_matcher(nlp):
    config = configparser.ConfigParser()
    config.read("paths.cfg")

    matcher = Matcher(nlp.vocab)
    for concept in cpnet_vocab:
        matcher.add(concept, None, [{"LEMMA": concept}])

    return matcher


def grounding_sentences(src, tgt, labels, type):
    # nlp.add_pipe(nlp.create_pipe('sentencizer'))
    res, error_ids = match_mentioned_concepts(question=src, answer=tgt, labels=labels)
    # res: "sent": s, "ans": a, "qc": list(question_concepts), "ac": list(answer_concepts)}
    with open( "{}_concepts_nv_star.json".format(type), 'w') as f:
        for line in res:
            json.dump(line, f)
            f.write('\n')

def read_csv_src(data_path, clo_prem):
    label2idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    data = []
    labels = []
    expl = []
    with open(data_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            if i>=1:
                data.append(row[clo_prem])
                labels.append(label2idx[row[1:2][0]])
                expl.append(row[4])
    return data, labels, expl

def read_csv_tgt(data_path, clo_hypo):
    data = []
    with open(data_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            if i>=1:
                hypo = row[clo_hypo]
                data.append(hypo)
    return data

def read_json(data_type):
    data_source, data_target, data_label = [],[],[]
    f_path = '../data/snli/'+data_type+'_hints.json'
    with open(f_path, 'r') as our_hints:
        hint_lines = our_hints.readlines()
        try:
            for hint_line in hint_lines:
                hint_dic = json.loads(hint_line.replace('\\',''))
                prem_star = hint_dic['Premise']
                hypo_star = hint_dic['Hypothesis']
                label = hint_dic['Label']
                data_source.append(prem_star)
                data_target.append(hypo_star)
                data_label.append(label)
        except: print(hint_line)
    return  data_source, data_target, data_label


def read_model_vocab(data_path):
    global model_vocab
    vocab_dict = json.loads(open(data_path, 'r').readlines()[0])
    model_vocab = []
    for tok in vocab_dict.keys():
        model_vocab.append(tok[1:])
    print(len(model_vocab))

read_model_vocab(config["paths"]["gpt2_vocab"])

type = 'dev'
src, tgt, labels = read_json(type)
grounding_sentences(src, tgt, labels, type)
type = 'test'
src, tgt, labels = read_json(type)
grounding_sentences(src, tgt, labels, type)
type = 'train'
src, tgt, labels = read_json(type)
grounding_sentences(src, tgt, labels, type)
