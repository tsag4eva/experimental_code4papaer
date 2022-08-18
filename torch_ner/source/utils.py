import csv
import hashlib
import json
import os
import re
import pickle
import sys
import time
import torch
from torch.autograd import Variable
from collections import Counter
from datetime import timedelta, datetime
from glob import iglob
from mmap import mmap, ACCESS_READ


import numpy as np
import yaml


def load_big_file(fp: str):
    """
    读取大文件
    :param fp:
    :return:  <class 'generator'>
    """
    with open(fp, "r", encoding="utf-8") as f:
        m = mmap(f.fileno(), 0, access=ACCESS_READ)
        tmp = 0
        for i, char in enumerate(m):
            if char == b"\n":
                yield m[tmp:i + 1].decode()
                tmp = i + 1


def load_file(fp: str, sep: str = None, name_tuple=None):
    """
    读取文件；
    :param fp:
    :param sep:
    :param name_tuple:
    :return:
    """
    with open(fp, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if sep:
            if name_tuple:
                return map(name_tuple._make, [line.strip().split(sep) for line in lines])
            else:
                return [line.strip().split(sep) for line in lines]
        else:
            return lines


def load_json_file(fp: str):
    """
    加载json文件
    :param fp:
    :return:
    """
    with open(fp, "r", encoding="utf-8") as f:
        return [json.loads(line.strip(), encoding="utf-8") for line in f.readlines()]


def save_json_file(list_data, fp):
    """
    保存json文件
    :param list_data:
    :param fp:
    :return:
    """
    with open(fp, "w", encoding="utf-8") as f:
        for data in list_data:
            json_str = json.dumps(data, ensure_ascii=False)
            f.write("{}\n".format(json_str))
        f.flush()


def load_csv(fp, is_tsv: bool = False):
    """
    加载csv文件为OrderDict()列表
    :param fp:
    :param is_tsv:
    :return:
    """
    dialect = 'excel-tab' if is_tsv else 'excel'
    with open(fp, "r", encoding='utf-8') as f:
        reader = csv.DictReader(f, dialect=dialect)
        return list(reader)


def load_csv_tuple(fp, name_tuple):
    """
    加载csv文件为指定类
    :param fp:
    :param name_tuple:
    :return:
    """
    return map(name_tuple._make, csv.reader(open(fp, "r", encoding="utf-8")))


def load_pkl(fp):
    """
    加载pkl文件
    :param fp:
    :return:
    """
    with open(fp, 'rb') as f:
        data = pickle.load(f)
        return data


def save_pkl(data, fp):
    """
    保存pkl文件，数据序列化
    :param data:
    :param fp:
    :return:
    """
    with open(fp, 'wb') as f:
        pickle.dump(data, f)


def load_yaml(fp):
    """
    加载yml文件
    :param fp:
    :return:
    """
    return yaml.load(open(fp, "r", encoding='utf-8'),
                     Loader=yaml.SafeLoader)


def save_yaml(data, fp):
    """
    持久化yml文件
    :param data:
    :param fp:
    :return:
    """
    yaml.dump(
        data,
        open(fp, "w", encoding="utf-8"),
        allow_unicode=True,
        default_flow_style=False)


def calculate_distance(vector1, vector2):
    """
    计算两个向量的余弦相似度
    :param vector1: 向量1
    :param vector2: 向量2
    :return:
    """
    cosine_distance = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))  # 余弦夹角
    # euclidean_distance = np.sqrt(np.sum(np.square(vector1 - vector2)))  # 欧式距离
    return cosine_distance


def split_data(data_set, batch_size):
    """
    数据集切分
    :param data_set:
    :param batch_size:
    :return:
    """
    batch_list = []
    data_size = len(data_set)
    count = (data_size + batch_size - 1) // batch_size
    for i in range(count):
        last_index = data_size if (i + 1) * batch_size > data_size else (i + 1) * batch_size
        res = data_set[i * batch_size:last_index]
        batch_list.append(res)
    return batch_list


def get_time_dif(start_time):
    """
    获取已使用时间
    :param start_time: time.time()
    :return:
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def format_data(t: datetime):
    """
    时间格式化，time.strftime("%Y-%m-%d %H:%M:%S")
    :param t:
    :return:
    """
    return t.strftime("%Y-%m-%d %H:%M:%S")

class Logger(object):
    def __init__(self, filename='BERT-BLSTM-CRF.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def md5(s: str):
    """
    MD5加密
    :param s:
    :return:
    """
    m = hashlib.md5()
    m.update(s.encode("utf-8"))
    return m.hexdigest()

def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub("\d", "0", s)

def scan_fp(ph):
    """
    递归返回指定目录下的所有文件
    :param ph:
    :return:
    """
    path_list = []
    for p in os.listdir(ph):
        fp = os.path.join(ph, p)
        if os.path.isfile(fp):
            path_list.append(fp)
        elif os.path.isdir(fp):
            path_list.extend(scan_fp(fp))
    return path_list


def scan_fp_iglob(ph):
    """
    返回指定目录下的所有文件
    :param ph:
    :return:
    """
    return list(filter(lambda x: os.path.isfile(x), iglob(f"{ph}/**", recursive=True)))

def write_Fdata_list(Fdata_list):
    '''
    将F1值列表写入文件保存
    Args:
        Fdata_list: 训练过程中保存的F1值列表

    Returns:

    '''
    # for software in cpedatasave_dict:
    #     cpedatasave_dict[software] = sorted(cpedatasave_dict[count])
    with open('Fscore_data_list.py', 'w') as f_write:
        f_write.write('Fdata_list = ' + str(Fdata_list))

def write_lossdata_list(lossdata_list,model):
    if model == 0:
        with open('loss_data_list.py', 'w') as f_write:
            f_write.write('lossdata_list = ' + str(lossdata_list))
    elif model == 1:
        with open('testset_loss_data_list.py', 'w') as f_write:
            f_write.write('lossdata_list = ' + str(lossdata_list))

def element_by_element(x, y):
    assert torch.is_tensor(x)
    assert torch.is_tensor(y)
    return torch.mul(x, y)

def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    从字典创建映射（item to ID / ID to item）。项目按频率递减排序。
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    创建一个字典和单词映射，按照词频排序
    """
    words = [x[0].lower() if lower else x[0] for s in sentences for x in s]
    dico = dict(Counter(words))#统计词频
    # print(dico)

    dico['<PAD>'] = 10000001
    dico['<UNK>'] = 10000000
    dico = {k:v for k,v in dico.items() if v>=3}
    # print(dico)
    word_to_id, id_to_word = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), len(words)
    ))
    return dico, word_to_id, id_to_word


def char_mapping(sentences):
    """
    sentence形式：一个列表储存一个完整的句子，句子中每个单词又单独与其标签构成一个列表
    [['The', 'DT', 'I-NP', 'O'], ['case', 'NN', 'I-NP', 'O'],...]
    """
    chars = ''.join([w[0] for s in sentences for w in s])
    dico = dict(Counter(chars))
    dico['<PAD>'] = 10000001
    dico['<UNK>'] = 10000000

    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique characters" % len(dico))
    return dico, char_to_id, id_to_char

def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [word[-1] for s in sentences for word in s]
    dico = dict(Counter(tags)) #Counter是一种类似字典结构的对象
    # print(dico)
    dico['<START>'] = -1
    dico['<STOP>'] = -2
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag

def cap_feature(s):
    """
    Capitalization feature:
    1 = low caps
    2 = all caps
    3 = first letter caps
    4 = one capital (not first letter)
    标注大小写特征
    """
    if s.lower() == s:
        return 1
    elif s.upper() == s:
        return 2
    elif s[0].upper() == s[0]:
        return 3
    else:
        return 4

def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([line.rstrip().split()[0].strip()
        for line in open(ext_emb_path, 'r', encoding='utf-8')])
    # print(pretrained)
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    # print(dictionary)
    return dictionary, word_to_id, id_to_word

def char2ids(sentences, char_to_id):
    for tokens in sentences:
        chars_ids = [[char_to_id[c if c in char_to_id else '<UNK>'] for c in w]for w in tokens]
    return chars_ids

def convert_chars(chars_ids, mode):
    if mode == "LSTM":
        chars_sorted = sorted(chars_ids, key=lambda p: len(p), reverse=True)
        d = {}
        for i, ci in enumerate(chars_ids):
            for j, cj in enumerate(chars_sorted):
                if ci == cj and not j in d and not i in d.values():
                    d[j] = i
                    continue
        chars_length = [len(c) for c in chars_sorted]
        char_maxl = max(chars_length)
        chars_mask = np.zeros((len(chars_sorted), char_maxl), dtype="int")
        for i, c in enumerate(chars_sorted):
            chars_mask[i, :chars_length[i]] = c
        chars_mask = Variable(torch.LongTensor(chars_mask))
    if mode == "CNN":
        d = {}
        chars_length = [len(c) for c in chars_ids]  # chars有长有短，统计各个char的长度
        char_maxl = max(chars_length) # 找出最长的char做为padding的依据
        chars_mask = np.zeros((len(chars_length), char_maxl), dtype="int") # 以最长的char为长度创建一个用0填充的数组
        # 将chars对应数据添加到数组中，相当于对char做了padding
        for i, c in enumerate(chars_ids):
            chars_mask[i, :chars_length[i]] = c
        chars_mask = Variable(torch.LongTensor(chars_mask))

    return chars_mask, d

#计算tfidf特征矩阵
def tfidf_calculate(tv, sg, test_doc):
    tfidf_list = []

    test_sg = sg.tokenize_no_space(test_doc[0].lower())

    test_fit = tv.transform(test_doc)
    testvocab_name = tv.get_feature_names()
    testvocab_VSMarray = test_fit.toarray()
    vocabdict = dict(zip(testvocab_name, testvocab_VSMarray[0]))
    # 去除字典中值为0的元素
    for v in list(vocabdict.keys()):   #对字典a中的keys，相当于形成列表list
        if vocabdict[v] == 0:
            del vocabdict[v]

    for word in test_sg:
        for k in list(vocabdict.keys()):
            if k == word:
                tfidf_list.append(vocabdict[k])
                flag = 1
                break
            else:
                flag = 0
    if flag == 0:
        tfidf_list.append(0) #若为停用词，直接将其tfidf值设为0

    tmpsort_list = list(tfidf_list) #将tfidf列表的元素暂存，防止排序操作影响原列表顺序
    tfidf_feature = list(tfidf_list)
    tmpsort_list.sort(reverse=True) #对tfidf值进行降序排序
    # print(tmpsort_list)

    #将tfidf值由大到小排名的前五个值分别标记为1~5,并还原到原来的值的位置
    for i,v1 in enumerate(tmpsort_list):
        for j,v2 in enumerate(tfidf_feature):
            if v1 == v2 and i <= 4:
                tfidf_feature[j] = i+1
                break

    # 将其余的值标记为0
    for i,v in enumerate(tfidf_feature):
        if isinstance(v,float):
            tfidf_feature[i] = 0
        else:pass

    return tfidf_feature