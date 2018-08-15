import argparse
import pandas as pd
import json
import re
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import jieba
import jieba.posseg as pseg
import pickle
import os

DICT_DIR = 'dict'
DATA_DIR = 'data'

def parse_args():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_fname', type=str, default='item_storeid_50w.json', help='')
    parser.add_argument('--out_fname', type=str, default='out.csv', help='')
    args = parser.parse_args()

def clean(ustring):
    # 全形轉半形
    ustring = ustring.lower()
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar) # 轉unicode
        if inside_code == 12288:                              #全形空格直接轉換            
            inside_code = 32 
        elif (inside_code >= 65281 and inside_code <= 65374): # 轉換其他全形字
            inside_code -= 65248
        rstring += chr(inside_code)
    rstring = re.sub(r"\[([^]]+)\]|\(([^\)]+)\)|\【([^】]+)\】", "", rstring)
    rstring = re.sub(r"[^\w\s]+", "-", rstring)
    rstring = re.sub(r"[a-zA-Z\-]+[\d]+[a-zA-Z\-]+|[a-zA-Z\-]+[\d]+|[\d]+[a-zA-Z\-]+|[\d\-]+|[\s\-]*[\d]+[a-zA-Z\-\d\s]*$", " ", rstring)
#     rstring = re.sub(r"[a-zA-Z,.:;@#?!&$%^*()\-=+`~{}\[\]\\\/><|\"\']+[\d]+|[\d]+[a-zA-Z,.:;@#?!&$%^*()\-=+`~{}\[\]\\\/><|\"\']+|[,.:;@#?!&$%^*()\-=+`~{}\[\]\\\/><|\"\'\d]+|[\d]+[a-zA-Z0-9,.:;@#?!&$%^*()\-=+`~{}\[\]\\\/><|\"\'\s]+$", " ",rstring)
    rstring = re.sub(r"[\s]+", " ", rstring)
    return rstring
    
def term_extracter(content, mode = 'lcut'):
    ### three mode for segmentation
    if mode == 'lcut':
#         seg_list = jieba.lcut(content, cut_all=True) ## 全模式
        seg_list = jieba.lcut(content, cut_all = False) ## 精確模式
    elif mode == 'lcut_for_search':
        seg_list = jieba.lcut_for_search(content) ## 搜尋字典模式
    seg_list = [x for x in seg_list if x not in STOP_WORDS]
    return seg_list

def tag_extract(content):
    def find_freq(term):
        freq = 0
        try:
            freq = WORD_FREQ[term]
        except:
            pass
        return freq
    seg_list = pseg.lcut(content)
    
    n_list = []
    x_list = []
    
    prev_v_or_a = False
    temp_term = ''
    thresh = 50
    for idx, (term, term_type) in enumerate(seg_list):
        if term in STOP_WORDS:
            temp_term = ''
            continue
        if prev_v_or_a:
            prev_v_or_a = False
            if term_type == 'n':
                temp_term += term
                if find_freq(temp_term) > thresh:
                    n_list.append(temp_term)
                else:
                    n_list.append(term)
                temp_term = ''
                continue
            else:
                temp_term = ''
        if term_type == 'n':
            n_list.append(term)
        elif term_type in ('nz', 'nr'):
            if find_freq(term) > thresh:
                x_list.append(term)
        elif term_type == 'v':
            temp_term += term
            prev_v_or_a = True
        elif term_type == 'x':
            x_list.append(term)
        
    if len(n_list) > 1:
        out_terms = n_list
    elif len(n_list) == 1:
        if len(n_list[0]) == 1:
            if len(x_list) > 2:
                freq_of_terms = [find_freq(x) for x in x_list]
                tmp = sorted(zip(x_list, freq_of_terms), key = lambda x: x[1], reverse=True)
                out_terms = [x for (x, times) in tmp[:3]]
            elif len(x_list) > 0:
                out_terms = x_list
            else: 
                out_terms = n_list
        else:
            out_terms = n_list
    else:
        if len(x_list) > 2:
            freq_of_terms = [find_freq(x) for x in x_list]
            tmp = sorted(zip(x_list, freq_of_terms), key = lambda x: x[1], reverse=True)
            out_terms = [x for (x, times) in tmp[:3]]
        else:
            out_terms = x_list
            
    out_terms = set(out_terms)
    out = '/'.join(out_terms)
    return out

def main():
    
    ##### set dictionary and load stop words #####
    jieba.set_dictionary(os.path.join(DICT_DIR, "userdict.txt"))
    jieba.load_userdict(os.path.join(DICT_DIR, "dict.txt.big"))

    stop_words_file = 'stop_words.txt'
    with open(os.path.join(DICT_DIR, stop_words_file)) as f:
        doc = f.read()
    global STOP_WORDS
    STOP_WORDS = doc.split('\n')
    
    ##### load data #####
    data = pd.read_json(os.path.join(DATA_DIR, args.in_fname), lines=True)
#     r = range(474851,474900)
#     data = data.iloc[r]#######################################################################
    
    ##### preprocessing #####
    # remove punctuation and numbers
    data['new_desc'] = data['description'].apply(lambda s: clean(s))
    
    # token and get term freq
#     data['token'] = data['new_desc'].apply(lambda s: " ".join(term_extracter(s, 'lcut')))
    data['token_search'] = data['new_desc'].apply(lambda s: " ".join(term_extracter(s, 'lcut_for_search')))
    vec = CountVectorizer(token_pattern=r'[^\s]+')
    tf = vec.fit_transform(data['token_search'])
    sum_words = tf.sum(axis=0) 
    global WORD_FREQ
    WORD_FREQ = dict()
    for word, idx in vec.vocabulary_.items():
        WORD_FREQ[word] = sum_words[0, idx]
        
    ##### extract tags #####
    data['tags'] = data['new_desc'].apply(lambda s: tag_extract(s))
    
    ##### save result #####
    out_df = data[['description', 'storeID', 'tags']]
    out_df.to_csv(os.path.join(args.out_fname),index=0)

if __name__ == '__main__':
    
    parse_args()
    main()