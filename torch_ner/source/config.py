# -*- coding: utf-8 -*-
# @description: 
# @author:
# @time: 2022/2/10 19:49
# @file: config.py

import datetime
import os
import threading


class Config(object):
    _instance_lock = threading.Lock()
    _init_flag = False

    def __init__(self):
        if not Config._init_flag:
            Config._init_flag = True
            root_path = str(os.getcwd()).replace("\\", "/")
            if 'source' in root_path.split('/'):
                #self.base_path=torch_ner文件夹路径
                self.base_path = os.path.abspath(os.path.join(os.path.pardir))
            else:
                self.base_path = os.path.abspath(os.path.join(os.getcwd(), 'torch_ner'))
            self._init_train_config()

    def __new__(cls, *args, **kwargs):
        """
        单例类
        :param args:
        :param kwargs:
        :return:
        """
        if not hasattr(Config, '_instance'):
            with Config._instance_lock:
                if not hasattr(Config, '_instance'):
                    Config._instance = object.__new__(cls)
        return Config._instance

    def _init_train_config(self):
        self.label_list = []
        self.use_gpu = True
        self.device = "cpu"
        self.sep = " "
        self.zeros = True
        self.all_emb = True
        # default代表使用tokenizer.convert_tokens_to_ids转换tokens
        # glove代表使用glove6b预训练嵌入转换tokens
        # self.convert_tokens_mode = "default"



        # 输入数据集、输出目录
        self.train_file = os.path.join(self.base_path, 'data', 'train.txt')
        self.eval_file = os.path.join(self.base_path, 'data', 'dev.txt')
        self.test_file = os.path.join(self.base_path, 'data', 'test.txt')
        self.pre_emb_file = os.path.join(self.base_path, 'data', 'glove.6B.100d.txt')
        self.log_path = os.path.join(self.base_path, 'output', "logs")
        self.output_path = os.path.join(self.base_path, 'output', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

        # 指定Config, Tokenizer和Model三个核心模型的位置（就是从抱抱脸下载的那几个文件）
        self.model_name_or_path = os.path.join(self.base_path, 'bert-base-cased')

        # 以下是模型训练参数
        self.do_train = True
        self.do_eval = True
        self.do_test = False
        self.clean = True
        self.need_birnn = True
        self.do_lower_case = True
        self.rnn_dim = 128
        self.cap_embedding_dim = 128
        self.max_seq_length = 128
        self.train_batch_size = 24
        self.eval_batch_size = 24
        self.num_train_epochs = 20
        self.gradient_accumulation_steps = 1
        self.learning_rate = 3e-5
        self.adam_epsilon = 1e-8
        self.warmup_steps = 0
        self.logging_steps = 500
