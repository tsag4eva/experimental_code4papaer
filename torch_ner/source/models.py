import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from torchcrf import CRF


class BERT_BiLSTM_CRF(BertPreTrainedModel):

    def __init__(self, config, need_birnn=False, rnn_dim=128):
        super(BERT_BiLSTM_CRF, self).__init__(config)
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # out_dim:768
        out_dim = config.hidden_size

        if need_birnn:
            self.need_birnn = need_birnn
            # self.cap_embedding = nn.Linear(in_features=128, out_features=128)
            # LSTM(768, 128, batch_first=True, bidirectional=True)config.cap_embedding_dim
            self.birnn = nn.LSTM(input_size=config.hidden_size,
                                 hidden_size=rnn_dim, num_layers=1, bidirectional=True,
                                 batch_first=True)
            out_dim = rnn_dim * 2# 256
        else:
            self.need_birnn = need_birnn

        # Linear(in_features=256, out_features=15, bias=True)
        self.hidden2tag = nn.Linear(in_features=out_dim, out_features=config.num_labels)

        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self, input_ids, tags, token_type_ids=None, attention_mask=None):
        """
        BERT_BiLSTM_CRF模型的正向传播函数

        :param input_ids:      torch.Size([batch_size,seq_len]), 代表输入实例的tensor张量
        :param token_type_ids: torch.Size([batch_size,seq_len]), 一个实例可以含有两个句子,相当于标记
        :param attention_mask:     torch.Size([batch_size,seq_len]), 指定对哪些词进行self-Attention操作
        :param tags:
        :return:
        """
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        if self.need_birnn:
            # fusion embedding
            # cap_embedding_out = self.cap_embedding(caps_ids.float())
            # fusion_embedding = torch.cat((sequence_output, caps_ids), 1)
            # lstm_output, _ = self.birnn(fusion_embedding)
            sequence_output, _ = self.birnn(sequence_output)

        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        loss = -1 * self.crf(emissions, tags, mask=attention_mask.byte())

        return loss

    def predict(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        模型预测
        :param input_ids:
        :param token_type_ids:
        :param attention_mask:
        :return:
        """
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        return self.crf.decode(emissions, attention_mask.byte())
