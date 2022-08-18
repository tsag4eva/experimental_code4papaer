# -*- coding: utf-8 -*-
# @description: 
# @author:
# @time: 2022/2/10 20:32


import logging
import os

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

from torch_ner.source.config import Config
from torch_ner.source.logger import logger as logging
from torch_ner.source.utils import load_pkl, load_file, save_pkl, \
     word_mapping, cap_feature


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, sentence, label=None):
        self.guid = guid
        self.text = text
        self.sentence = sentence
        self.label = label



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, token_type_ids, attention_mask,
                caps_ids, label_id, ori_tokens):
        """
        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :param label_id:
        :param ori_tokens:
        """
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        # self.tokens_ids = tokens_ids
        # self.chars_ids = chars_ids
        self.caps_ids = caps_ids
        self.label_id = label_id
        self.ori_tokens = ori_tokens
        # self.dico_words = dico_words


class NerProcessor(object):
    def __init__(self):
        # 初始化系统配置、数据预处理
        self.config = Config()

    def load_examples(self, config: Config, mode="train"):
        if mode == "train":
            file_path = config.train_file
        elif mode == "dev":
            file_path = config.eval_file
        elif mode == "test":
            file_path = config.test_file
        else:
            raise ValueError("mode must be one of train, dev, or test")
        sentences = []
        # 读取输入数据，进一步封装
        examples = self.get_input_examples(file_path, separator=config.sep)


        for i, example in enumerate(tqdm(examples, desc=f"extract token statistics in {mode}")):
            example_sentence_list = example.sentence
            sentences.append(example_sentence_list)

        dico_tokens = word_mapping(sentences, lower=True)[0]

        return examples, dico_tokens, sentences

    def get_dataset(self, config: Config, tokenizer, examples, token_to_id=0, mode='default'):
        """
        对指定数据集进行预处理，进一步封装数据，包括:
        examples：[InputExample(guid=index, text=text, label=label)]
        features：[InputFeatures( input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask,
                                  label_id=label_ids,
                                  ori_tokens=ori_tokens)]
        data： 处理完成的数据集, TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_label_ids)

        :param config:
        :param tokenizer:
        :param mode:
        :return:
        """

        # 对输入数据进行特征转换
        features = self.convert_examples_to_features(config, examples, tokenizer, token_to_id, mode)


        # 获取全部数据的特征，封装成TensorDataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        # all_tokens_ids = torch.tensor([f.tokens_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        if mode == "glove":
            all_caps_ids = torch.tensor([f.caps_ids for f in features], dtype=torch.long)
            data = TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask,
                            all_caps_ids, all_label_ids)  # all_chars_ids
        else:
            data = TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask,
                                all_label_ids)


        return examples, features, data

    @staticmethod
    def convert_examples_to_features(config: Config, examples, tokenizer, token_to_id, mode):
        """
        对输入数据进行特征转换
        :param config:
        :param examples:
        :param tokenizer:
        :return:
        """
        lower = True
        def f(x): return x.lower() if lower else x

        label_map = {label: i for i, label in enumerate(config.label_list)}
        max_seq_length = config.max_seq_length
        features = []
        for ex_index, example in enumerate(tqdm(examples, desc="convert examples")):
            example_text_list = example.text.split(" ")
            example_label_list = example.label.split(" ")
            # example_sentence_list = example.sentence
            tokenlist, token, tokens, labels, ori_tokens = [], [], [], [], []
            word_piece = False

            assert len(example_text_list) == len(example_label_list)
            # assert len(example_text_list) == len(example_sentence_list)

            for i, word in enumerate(example_text_list):
                # 防止wordPiece情况出现，不过貌似不会
                # 不使用tokenize对数据集提供的token再做分词
                # 因为用tokenizer.tokenize处理单个英文单词时会默认使用wordPiece
                # token = tokenizer.tokenize(word)
                token = word
                tokenlist.append(word)# 用于计算单词个数避免单词被分成字母
                tokens.append(token)
                label = example_label_list[i]
                ori_tokens.append(word)
                # 单个字符不会出现wordPiece
                if len(tokenlist) == 1:
                    labels.append(label)
                    tokenlist = []
                else:
                    word_piece = True
                    tokenlist = []

            if word_piece:
                logging.info("Error tokens!!! skip this lines, the content is: %s" % " ".join(example_text_list))
                continue

            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]
                ori_tokens = ori_tokens[0:(max_seq_length - 2)]
            ori_tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]
            new_tokens = ["[CLS]"] + tokens + ["[SEP]"]

            if mode == "glove":
                input_ids = [token_to_id[f(w) if f(w) in token_to_id else '<UNK>'] for w in tokens]
                # chars_ids = [[char_to_id[c if c in char_to_id else '<UNK>'] for c in w]for w in tokens]
                input_ids = [label_map["O"]] + input_ids + [label_map["O"]]

            else:
                assert len(ori_tokens) == len(new_tokens)
                input_ids = tokenizer.convert_tokens_to_ids(new_tokens)

            caps_ids = [cap_feature(w) for w in tokens] #遍历str_words获取单个词语，然后使用cap_feature识别大小写特征
            caps_ids = [label_map["O"]] + caps_ids + [label_map["O"]]

            label_ids = [label_map[labels[i]] for i, token in enumerate(tokens)]


            label_ids = [label_map["O"]] + label_ids + [label_map["O"]]
            token_type_ids = [0] * len(input_ids)
            attention_mask = [1] * len(input_ids)

            # assert len(ori_tokens) == len(new_tokens)

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                attention_mask.append(0)
                token_type_ids.append(0)
                label_ids.append(0)
                # tokens_ids.append(0)
                caps_ids.append(0)
                new_tokens.append("*NULL*")
                # lower_tokens.append("*NULL*")

            assert len(input_ids) == max_seq_length
            assert len(attention_mask) == max_seq_length
            assert len(token_type_ids) == max_seq_length
            # assert len(tokens_ids) == max_seq_length
            assert len(caps_ids) == max_seq_length
            assert len(label_ids) == max_seq_length

            if ex_index < 3:
                logging.info("****** Example ******")
                logging.info("guid: %s" % example.guid)
                logging.info("tokens: %s" % " ".join([str(x) for x in new_tokens]))
                logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logging.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logging.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                # logging.info("tokens_ids: %s" % " ".join([str(x) for x in tokens_ids]))
                logging.info("caps_ids: %s" % " ".join([str(x) for x in caps_ids]))
                logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))


            features.append(InputFeatures(input_ids=input_ids,
                                          token_type_ids=token_type_ids,
                                          attention_mask=attention_mask,
                                          caps_ids=caps_ids,
                                          label_id=label_ids,
                                          ori_tokens=ori_tokens,
                                          ))
        return features

    def get_input_examples(self, input_file, separator=" "):
        """
        通过读取输入数据，封装输入样本
        :param separator:
        :param input_file:
        :return:
        """
        examples = []
        lines = self.read_data(input_file, separator=separator)
        for i, line in enumerate(lines):
            guid = str(i)
            text = line[1]
            label = line[0]
            sentence = line[2]
            examples.append(InputExample(guid=guid, text=text, sentence=sentence, label=label))
        return examples

    @staticmethod # tag_mapping
    def get_labels(config: Config):
        """
        读取训练数据获取标签
        :param config:
        :return:
        """
        label_pkl_path = os.path.join(config.output_path, "label_list.pkl")
        if os.path.exists(label_pkl_path):
            logging.info(f"loading labels info from {config.output_path}")
            labels = load_pkl(label_pkl_path)
        else:
            logging.info(f"loading labels info from train file and dump in {config.output_path}")
            tokens_list = load_file(config.train_file, sep=config.sep)
            labels = set([tokens[1] for tokens in tokens_list if len(tokens) == 2])

        if len(labels) == 0:
            ValueError("loading labels error, labels type not found in data file: {}".format(config.output_path))
        else:
            save_pkl(labels, label_pkl_path)

        return labels

    @staticmethod
    def get_label2id_id2label(output_path, label_list):
        """
        获取label2id、id2label的映射
        :param output_path:
        :param label_list:
        :return:
        """
        label2id_path = os.path.join(output_path, "label2id.pkl")
        if os.path.exists(label2id_path):
            label2id = load_pkl(label2id_path)
        else:
            label2id = {l: i for i, l in enumerate(label_list)}
            save_pkl(label2id, label2id_path)

        id2label = {value: key for key, value in label2id.items()}
        return label2id, id2label

    @staticmethod
    def read_data(input_file, separator="\t", zeros=True):
        """
        读取输入数据
        :param input_file:
        :param separator:
        :return:
        """
        count = 0
        with open(input_file, "r", encoding="utf-8") as f:
            lines, words, labels, sentence, sentences = [], [], [], [], []
            for line in f.readlines():
                contends = line.strip()
                tokens = line.strip().split(separator)
                # 新增
                if not contends and len(sentence) > 0:
                    if 'DOCSTART' not in sentence[0][0]:
                        sentences.append(sentence) #遇到截止标记，将之前获取的一条句子加到sentences
                        count += 1
                    sentence = []
                else:
                    assert len(tokens) >= 2
                    sentence.append(tokens)
                #

                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[1])
                else:
                    if len(contends) == 0 and len(words) > 0:
                        label = []
                        word = []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word), sentences[count-1]])
                        # lines.append(sentences[count-1])
                        words = []
                        labels = []
                        # print(sentences[count-1])
            # 新增
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence) #遇到截止标记，将之前获取的一条句子加到sentences
            #

            return lines

    @staticmethod
    def clean_output(config: Config):
        """
        清理output_xxx目录，若output_xxx目录存在，将会被删除, 然后初始化输出目录
        :param config:
        :return:
        """
        if config.clean and config.do_train:
            logging.info(f"clear output dir: {config.output_path}")
            if os.path.exists(config.output_path):
                def del_file(path):
                    ls = os.listdir(path)
                    for i in ls:
                        c_path = os.path.join(path, i)
                        if os.path.isdir(c_path):
                            del_file(c_path)
                            os.rmdir(c_path)
                        else:
                            os.remove(c_path)

                try:
                    del_file(config.output_path)
                except Exception as e:
                    logging.error(e)
                    logging.error('pleace remove the files of output dir and data.conf')
                    exit(-1)

        # 初始化output目录
        if os.path.exists(config.output_path) and os.listdir(config.output_path) and config.do_train:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(config.output_path))

        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        if not os.path.exists(os.path.join(config.output_path, "dev")):
            os.makedirs(os.path.join(config.output_path, "dev"))




