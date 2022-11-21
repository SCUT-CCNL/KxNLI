import spacy
import configparser
import json
import re
import csv
import collections
import math
from nltk.translate.bleu_score import sentence_bleu

concept_vocab = set()
config = configparser.ConfigParser()
config.read("../preprocess/paths.cfg")
with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
    cpnet_vocab = [l.strip() for l in list(f.readlines())]
cpnet_vocab = set([c.replace("_", " ") for c in cpnet_vocab])
def read_model_vocab(data_path):
    global model_vocab
    vocab_dict = json.loads(open(data_path, 'r').readlines()[0])
    model_vocab = []
    for tok in vocab_dict.keys():
        model_vocab.append(tok[1:])

read_model_vocab(config["paths"]["gpt2_vocab"])

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

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
def hard_ground(sent):
    global cpnet_vocab, model_vocab
    sent = sent.lower()
    doc = nlp(sent)
    res = set()
    for t in doc:
        if t.lemma_ in cpnet_vocab and t.lemma_ not in blacklist \
                and t.lemma_ in model_vocab: 
            res.add(t.lemma_)
    return res

def cpt_num(sent):
    res = []
    all_concepts = hard_ground(sent)
    num_cpts = len(all_concepts)
    return num_cpts

def readNLE_cpts(fname):
    total_num = 0
    num =0
    with open(fname, 'r') as NLE:
        lines = NLE.readlines()
        for line in lines:
            num += 1
            total_num +=cpt_num(line)/len(line)
    return total_num/num

def read_golden_rationales(gen_f, golden_f, rationale_f):

    num = 0
    tf = 0
    with open(golden_f, 'r') as csv_file,\
            open(gen_f, 'r') as gen_Ra, \
            open(rationale_f, 'r') as Ra:
        csv_reader = csv.reader(csv_file, delimiter=',')
        gen_lines = gen_Ra.readlines()
        ra_lines = Ra.readlines()
        for row_count, row in enumerate(csv_reader):
            if row_count == 0:
                continue
            num += 1
            golden_sent = row[5] + row[6]
            golden_sent = golden_sent.lower()  
            golden_ra_list = only_hints(golden_sent)
            list2seq = ' '.join(golden_ra_list)
            golden_ra_list = nlp(list2seq)
            golden_ra_list = [g.lemma_ for g in golden_ra_list]
            # if '.csv' in rationale_f:
            #     gen_Ra = row[4].split()
            # else:
            gen_Ra = nlp(gen_lines[row_count-1])
            # input_rationale = only_hints(ra_lines[row_count-1].replace('[ ','*').replace(' ]','*').replace('[ . ]','.').replace('[ , ]','.'))
            NLE_rationale = [j for j in gen_Ra if j.lemma_ in golden_ra_list]
            tf += len(NLE_rationale)/len(gen_Ra)
    return tf/num

def only_hints(sent):
    pattern = re.compile(r'\*(.*?)\*')
    w = pattern.findall(sent)
    w = [i.replace('.','') for i in w]
    return w

def get_bleu(ref, gen):
    bleu_1 = 0
    bleu_2 = 0
    bleu_3 = 0
    bleu_4 = 0
    with open(ref, 'r') as csv_file,\
            open(gen, 'r') as gen_NLE:
        csv_reader = csv.reader(csv_file, delimiter=',')
        gen_lines = gen_NLE.readlines()
        for row_count, row in enumerate(csv_reader):
            if row_count == 0:
                continue
            golden_NLE_list = [row[4].split()]
            gen_NLE_list = gen_lines[row_count-1].split()
            bleu_1 += sentence_bleu(golden_NLE_list, gen_NLE_list, weights=(1, 0, 0, 0))
            bleu_2 += sentence_bleu(golden_NLE_list, gen_NLE_list, weights=(0, 1, 0, 0))
            bleu_3 += sentence_bleu(golden_NLE_list, gen_NLE_list, weights=(0, 0, 1, 0))
            bleu_4 += sentence_bleu(golden_NLE_list, gen_NLE_list, weights=(0, 0, 0, 1))
    return bleu_1/row_count, bleu_2/row_count, bleu_3/row_count, bleu_4/row_count

def eval_bleu(data_type):
    golden_f = '../data/snli/'+data_type+'/source.csv'
    # gen_f_our = '../models/snli/att-medium-withR-snli_eval/myR/20hop_result_ep:' + data_type + '.txt'  # 25842 0.279 0.137 0.075 0.042

    gen_f_our = '../models/snli/att-medium-withR-snli_eval/1111111111_'+data_type+'.txt'  # 25842 0.279 0.137 0.075 0.042
    gen_f_lirex = '../models/snli/att-medium-withR-snli_eval/lirex/lirex_'+data_type+'.txt'  # 21069 0.219 0.098 0.049 0.024
    gen_f_golden = '../models/snli/att-medium-withR-snli_eval/golden/'+data_type+'_withE.txt'  # 31232 0.373, 0.215 0.134 0.087
    gen_f_diff = '../models/snli/att-medium-withR-snli_eval/diff/15/20hop_result_ep:'+data_type+'.txt'  # (0.2482868268090833, 0.11256835222343428, 0.05648006439220019, 0.028650952746793006)
    gen_f_noKG = '../models/snli/withRational-noKG-snli_eval/noKG_'+data_type+'.txt'
    gen_f_noR = '../models/snli/att-medium-withoutR-snli_eval/20hop_result_ep:' + data_type + '.txt'

    print('Our: ', get_bleu(golden_f, gen_f_our))
    print('lirex: ', get_bleu(golden_f, gen_f_lirex))
    print('diff: ', get_bleu(golden_f, gen_f_diff))
    print('golden: ', get_bleu(golden_f, gen_f_golden))
    print('noKG: ', get_bleu(golden_f, gen_f_noKG))
    print('noR: ', get_bleu(golden_f, gen_f_noR))

def eval_rationale(data_type):
    golden_f = '../data/snli/' + data_type + '/source.csv'
    gen_f_our = '../models/snli/att-medium-withR-snli_eval/myR/20hop_result_ep:' + data_type + '.txt'  # 25842 0.279 0.137 0.075 0.042
    # gen_f_our = '../models/snli/att-medium-withR-snli_eval/1111111111_' + data_type + '.txt'  # 25842 0.279 0.137 0.075 0.042
    gen_f_lirex = '../models/snli/att-medium-withR-snli_eval/lirex/lirex_' + data_type + '.txt'  # 21069 0.219 0.098 0.049 0.024
    gen_f_golden = '../models/snli/att-medium-withR-snli_eval/golden/' + data_type + '_withE.txt'  # 31232 0.373, 0.215 0.134 0.087
    gen_f_diff = '../models/snli/att-medium-withR-snli_eval/diff/15/20hop_result_ep:' + data_type + '.txt'  # (0.2482868268090833, 0.11256835222343428, 0.05648006439220019, 0.028650952746793006)
    gen_f_noKG = '../models/snli/withRational-noKG-snli_eval/noKG_'+data_type+'.txt'
    gen_f_noR = '../models/snli/att-medium-withoutR-snli_eval/20hop_result_ep:' + data_type + '.txt'

    rationales_our = '../data/snli/' + data_type + '/source_with_my_hints.json'
    rationales_noKG = '../data/snli/' + data_type + '/source_with_my_hints.json'
    rationales_noR = '../data/snli/' + data_type + '/source_with_my_hints.json'
    rationales_lirex = '../data/snli/' + data_type + '/source_with_lirex_hints.json'
    rationales_diff = '../data/snli/' + data_type + '/source_with_diff_hints15.txt'
    rationales_golden = '../data/snli/' + data_type + '/source.csv'


    print('Our: ', read_golden_rationales(gen_f_our, golden_f, rationales_our))
    print('lirex: ', read_golden_rationales(gen_f_lirex, golden_f, rationales_lirex))
    print('diff: ', read_golden_rationales(gen_f_diff, golden_f, rationales_diff))
    print('noKG: ', read_golden_rationales(gen_f_noKG, golden_f, rationales_noKG))
    print('noR: ', read_golden_rationales(gen_f_noR, golden_f, rationales_noR))
    print('golden: ', read_golden_rationales(gen_f_golden, golden_f, rationales_golden))

def get_cpts(data_type):
    gen_f_our = '../models/snli/att-medium-withR-snli_eval/myR/20hop_result_ep:' + data_type + '.txt'  # 25842 0.279 0.137 0.075 0.042
    gen_f_lirex = '../models/snli/att-medium-withR-snli_eval/lirex/lirex_' + data_type + '.txt'  # 21069 0.219 0.098 0.049 0.024
    gen_f_golden = '../models/snli/att-medium-withR-snli_eval/golden/' + data_type + '_withE.txt'  # 31232 0.373, 0.215 0.134 0.087
    gen_f_diff = '../models/snli/att-medium-withR-snli_eval/diff/15/20hop_result_ep:' + data_type + '.txt'  # (0.2482868268090833, 0.11256835222343428, 0.05648006439220019, 0.028650952746793006)
    print('------------------------dev------------------------')
    print('golden\t', readNLE_cpts(fname=gen_f_golden))
    print('diff\t', readNLE_cpts(fname=gen_f_diff))
    print('lirex\t', readNLE_cpts(fname=gen_f_lirex))
    print('our\t', readNLE_cpts(fname=gen_f_our))

if __name__ == '__main__':
    print('----------------------BLEU--dev------------------------')
    eval_bleu(data_type = 'dev')
    print('----------------------BLEU--test------------------------')
    eval_bleu(data_type = 'test')

    print('----------------------RATIONALES--dev------------------------')
    eval_rationale(data_type='dev')
    print('----------------------RATIONALES--test------------------------')
    eval_rationale(data_type='test')
 
