# coding:utf-8
import codecs
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import json
from utils import load_vocab
# from ner_constant import *

# -*- coding: utf-8 -*-
import torch

class config:
    # tag-entity:{d:疾病 s:临床表现 b:身体 e:医疗设备 p:医疗程序 m:微生物类 k:科室 i:医学检验项目 y:药物}
    l2i_dic = {"o": 0, "d-B": 1, "d-M": 2, "d-E": 3, "s-B": 4, "s-M": 5,"s-E": 6,
               "b-B": 7, "b-M": 8, "b-E": 9, "e-B": 10, "e-M": 11, "e-E": 12, "p-B": 13, "p-M": 14, "p-E": 15, "m-B": 16,"m-M": 17,
               "m-E": 18, "k-B": 19, "k-M": 20, "k-E": 21, "i-B": 22, "i-M": 23,"i-E": 24, "y-B": 25, "y-M": 26, "y-E": 27,"<pad>":28,"<start>": 29, "<eos>": 30}

    i2l_dic = {0: "o", 1: "d-B", 2: "d-M", 3: "d-E", 4: "s-B", 5: "s-M",
               6: "s-E", 7: "b-B", 8: "b-M", 9: "b-E", 10: "e-B", 11: "e-M", 12: "e-E",13:"p-B", 14:"p-M", 15:"p-E",
               16: "m-B", 17: "m-M", 18: "m-E", 19: "k-B",20: "k-M", 21: "k-E",
              22: "i-B", 23: "i-M", 24: "i-E", 25: "y-B", 26: "y-M", 27: "y-E", 28: "<pad>",29:"<start>", 30:"<eos>"}


    # train_file = './data/train_data.txt'
    # dev_file = './data/val_data.txt'
    # test_file = './data/test_data.txt'
    # vocab_file = './data/my_bert/vocab.txt'

    # save_model_dir =  './data/model/'
    # medical_tool_model = './data/model/model.pkl'
    max_length = 450
    batch_size = 1
    epochs = 30
    tagset_size = len(l2i_dic)
    use_cuda = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ner_model = './model/medical_ner'
    newpath   = './model/medical_ner/model.pkl'
    vocabpath = './model/medical_ner/vocab.txt'



from model_ner import BERT_LSTM_CRF
import os


class medical_ner(object):
    def __init__(self,config ):
        self.config         = config
        self.NEWPATH        = config.newpath
        self.vocab          = load_vocab(config.vocabpath)
        self.vocab_reverse  = {v: k for k, v in self.vocab.items()}

        self.model          = BERT_LSTM_CRF(config.ner_model, config.tagset_size, 768, 200, 2,
                              dropout_ratio=0.5, dropout1=0.5, use_cuda=config.use_cuda)
        if config.use_cuda:
            self.model.to(config.device)
    def from_input(self, input_str):
        raw_text = []
        textid = []
        textmask = []
        textlength = []
        text = ['[CLS]'] + [x for x in input_str] + ['[SEP]']
        raw_text.append(text)
        cur_len = len(text)
        # raw_textid = [self.vocab[x] for x in text] + [0] * (max_length - cur_len)
        raw_textid = [self.vocab[x] for x in text if self.vocab.__contains__(x)] + [0] * (self.config.max_length - cur_len)
        textid.append(raw_textid)
        raw_textmask = [1] * cur_len + [0] * (self.config.max_length - cur_len)
        textmask.append(raw_textmask)
        textlength.append([cur_len])
        textid = torch.LongTensor(textid)
        textmask = torch.LongTensor(textmask)
        textlength = torch.LongTensor(textlength)
        return raw_text, textid, textmask, textlength

    def from_txt(self, input_path):
        raw_text = []
        textid = []
        textmask = []
        textlength = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if len(line.strip())==0:
                    continue
                if len(line) > 448:
                    line = line[:448]
                temptext = ['[CLS]'] + [x for x in line[:-1]] + ['[SEP]']
                cur_len = len(temptext)
                raw_text.append(temptext)

                tempid = [self.vocab[x] for x in temptext[:cur_len]] + [0] * (max_length - cur_len)
                textid.append(tempid)
                textmask.append([1] * cur_len + [0] * (max_length - cur_len))
                textlength.append([cur_len])

        textid = torch.LongTensor(textid)
        textmask = torch.LongTensor(textmask)
        textlength = torch.LongTensor(textlength)
        return raw_text, textid, textmask, textlength
    def split_entity_input(self,label_seq):
        entity_mark = dict()
        entity_pointer = None
        for index, label in enumerate(label_seq):
            #print(f"before: {label_seq}")
            if label.split('-')[-1]=='B':
                category = label.split('-')[0]
                entity_pointer = (index, category)
                entity_mark.setdefault(entity_pointer, [label])
            elif label.split('-')[-1]=='M':
                if entity_pointer is None: continue
                if entity_pointer[1] != label.split('-')[0]: continue
                entity_mark[entity_pointer].append(label)
            elif label.split('-')[-1]=='E':
                if entity_pointer is None: continue
                if entity_pointer[1] != label.split('-')[0]: continue
                entity_mark[entity_pointer].append(label)
            else:
                entity_pointer = None
           # print(entity_mark)
        return entity_mark
    def predict_sentence(self, sentence):
        tag_dic = {"d": "疾病", "b": "身体", "s": "症状", "p": "医疗程序", "e": "医疗设备", "y": "药物", "k": "科室",
                   "m": "微生物类", "i": "医学检验项目"}
        if sentence == '':
            print("输入为空！请重新输入")
            return
        if len(sentence) > 448:
            print("输入句子过长，请输入小于148的长度字符！")
            sentence = sentence[:448]
        raw_text, test_ids, test_masks, test_lengths = self.from_input(sentence)
        test_dataset = TensorDataset(test_ids, test_masks, test_lengths)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
        self.model.load_state_dict(torch.load(self.NEWPATH, map_location=self.config.device))
        self.model.eval()

        for i, dev_batch in enumerate(test_loader):
            sentence, masks, lengths = dev_batch
            batch_raw_text = raw_text[i]
            sentence, masks, lengths = Variable(sentence), Variable(masks), Variable(lengths)
            if self.config.use_cuda:
                sentence = sentence.to(self.config.device)
                masks = masks.to(self.config.device)

            predict_tags = self.model(sentence, masks)
            predict_tags.tolist()
            predict_tags = [self.config.i2l_dic[t.item()] for t in predict_tags[0]]
            predict_tags = predict_tags[:len(batch_raw_text)]
            pred = predict_tags[1:-1]
            raw_text = batch_raw_text[1:-1]
            entity_mark = self.split_entity_input(pred)
            entity_list = {}
            if entity_mark is not None:
                for item, ent in entity_mark.items():
                    # print(item, ent)
                    entity = ''
                    index, tag = item[0], item[1]
                    len_entity = len(ent)

                    for i in range(index, index + len_entity):
                        entity = entity + raw_text[i]
                    entity_list[tag_dic[tag]] = entity
            # print(entity_list)
        return entity_list
    def predict_file(self, input_file, output_file):
        tag_dic = {"d": "疾病", "b": "身体", "s": "症状", "p": "医疗程序", "e": "医疗设备", "y": "药物", "k": "科室",
                   "m": "微生物类", "i": "医学检验项目"}
        raw_text, test_ids, test_masks, test_lengths = self.from_txt(input_file)
        test_dataset = TensorDataset(test_ids, test_masks, test_lengths)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
        self.model.load_state_dict(torch.load(self.NEWPATH, map_location=device))
        self.model.eval()
        op_file = codecs.open(output_file, 'w', 'utf-8')
        for i, dev_batch in enumerate(test_loader):
            sentence, masks, lengths = dev_batch
            batch_raw_text = raw_text[i]
            sentence, masks, lengths = Variable(sentence), Variable(masks), Variable(lengths)
            if self.config.use_cuda:
                sentence = sentence.to(self.config.device)
                masks = masks.to(self.config.device)

            predict_tags = self.model(sentence, masks)
            predict_tags.tolist()
            predict_tags = self.model(sentence, masks)
            predict_tags.tolist()
            predict_tags = [self.config.i2l_dic[t.item()] for t in predict_tags[0]]
            predict_tags = predict_tags[:len(batch_raw_text)]
            pred = predict_tags[1:-1]
            raw_text = batch_raw_text[1:-1]

            entity_mark = self.split_entity_input(pred)
            entity_list = {}
            if entity_mark is not None:
                for item, ent in entity_mark.items():
                    entity = ''
                    index, tag = item[0], item[1]
                    len_entity = len(ent)
                    for i in range(index, index + len_entity):
                        entity = entity + raw_text[i]
                    entity_list[tag_dic[tag]] = entity
            op_file.write("".join(raw_text))
            op_file.write("\n")
            op_file.write(json.dumps(entity_list, ensure_ascii=False))
            op_file.write("\n")

        op_file.close()
        print('处理完成！')
        print("结果保存至 {}".format(output_file))


if __name__ == "__main__":
    sentence = "抑郁症受遗传的影响。在抑郁症青少年中，约25%～33%的家庭有一级亲属的发病史，是没有抑郁症青少年家庭发病的2倍。"
    my_pred = medical_ner(config)
    res = my_pred.predict_sentence(sentence)
    print("---")
    print(res)
