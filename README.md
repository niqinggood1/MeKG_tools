# CMeKG 工具 代码及模型


Index
---
<!-- TOC -->

- [CMeKG工具](#cmekg工具)
  - [模型下载](#模型下载)
- [依赖库](#依赖库)
- [模型使用](#模型使用)
  - [关系抽取](#医学关系抽取)
  - [医学实体识别](#医学实体识别)
  - [医学文本分词](#医学文本分词)


<!-- /TOC -->

原始代码：https://github.com/king-yyf/CMeKG_tools/tree/main  
## cmekg工具

[CMeKG网站](https://cmekg.pcl.ac.cn/)

中文医学知识图谱CMeKG
CMeKG（Chinese Medical Knowledge Graph）是利用自然语言处理与文本挖掘技术，基于大规模医学文本数据，以人机结合的方式研发的中文医学知识图谱。
CMeKG 中主要模型工具包括 医学文本分词，医学实体识别和医学关系抽取。这里是三种工具的代码、模型和使用方法。

### 模型下载

由于依赖和训练好的的模型较大，将模型放到了百度网盘中，链接如下，按需下载。

RE：链接:https://pan.baidu.com/s/1cIse6JO2H78heXu7DNewmg  密码:4s6k

NER: 链接:https://pan.baidu.com/s/16TPSMtHean3u9dJSXF9mTw  密码:shwh

分词：链接:https://pan.baidu.com/s/1bU3QoaGs2IxI34WBx7ibMQ  密码:yhek

## 依赖库

- json
- random
- numpy
- torch
- transformers
- gc
- re
- time
- tqdm

## 简述
medical_re.py  医学关系抽取  
medical_ner.py 命名实体识别  
medical_cws.py 医学分词(较少使用，可用jieba替代)  

## 模型使用

### 医学关系抽取
使用脚本 medical_re.py 

**依赖文件**
-  pytorch_model.bin : 医学文本预训练的 BERT-base model
-  vocab.txt
-  config.json
-  model_re.pkl: 训练好的关系抽取模型文件，包含了模型参数、优化器参数等
-  predicate.json 

**使用方法**

配置参数在medical_re.py的class config里，
首先在medical_re.py的class config里修改各个文件路径


- 使用
```
训练： 参考run_train    函数进行训练，其中model_re/train_example.json 是训练样本示例  
关系提取， 参考extract_data 函数
load_schema(PATH_SCHEMA)
model4s, model4po = load_model( PATH_SCHEMA,  PATH_MODEL )
text        = "右肺恶性肿瘤 患者男，63岁，因“右侧肺癌术后2年，气喘1周”入院。治疗过程：患者入院后完善相关检查，拟转本院南院行气管支架置入术"
res         = get_triples(text,tokenizer,max_seq_len,num_p,id2predicate,\
                              model4s,model4po)
print(res)
```
- 执行结果
```
[
 {
  "text": "据报道称，新冠肺炎患者经常会发热、咳嗽，少部分患者会胸闷、=乏力，其病因包括: 1.自身免疫系统缺陷\n2.人传人",
  "triples": [
   [
    "新冠肺炎",
    "临床表现",
    "肺炎"
   ],
   [
    "新冠肺炎",
    "临床表现",
    "发热"
   ],
   [
    "新冠肺炎",
    "临床表现",
    "咳嗽"
   ],
   [
    "新冠肺炎",
    "临床表现",
    "胸闷"
   ],
   [
    "新冠肺炎",
    "临床表现",
    "乏力"
   ],
   [
    "新冠肺炎",
    "病因",
    "自身免疫系统缺陷"
   ],
   [
    "新冠肺炎",
    "病因",
    "人传人"
   ]
  ]
 }
]
```

### 医学实体识别

**训练**
python3 train_ner.py

**使用进行提取**
参考 medical_ner.py 中的predict_sentence函数
medical_ner 类提供两个接口测试函数
- predict_sentence(sentence): 测试单个句子，返回:{"实体类别"：“实体”},不同实体以逗号隔开
- predict_file(input_file, output_file): 测试整个文件
文件格式每行待提取实体的句子和提取出的实体{"实体类别"：“实体”},不同实体以逗号隔开

```python
#使用工具运行
my_pred=medical_ner()
#根据提示输入单句：“高血压病人不可食用阿莫西林等药物”
sentence=input("输入需要测试的句子:")
my_pred.predict_sentence("".join(sentence.split()))

#输入文件(测试文件，输出文件)
my_pred.predict_file("my_test.txt","outt.txt")
```

### 医学文本分词(此处跳过)

调整的参数和模型在cws_constant.py中，意义不是特别大，用jieba分词更简单

**训练**
python3 train_cws.py
**使用示例**
medical_cws 类提供两个接口测试函数
- predict_sentence(sentence): 测试单个句子，返回:{"实体类别"：“实体”},不同实体以逗号隔开
- predict_file(input_file, output_file): 测试整个文件
文件格式每行待提取实体的句子和提取出的实体{"实体类别"：“实体”},不同实体以逗号隔开

```python
from run import medical_cws

#使用工具运行
my_pred=medical_cws()
#根据提示输入单句：“高血压病人不可食用阿莫西林等药物”
sentence=input("输入需要测试的句子:")
my_pred.predict_sentence("".join(sentence.split()))

#输入文件(测试文件，输出文件)
my_pred.predict_file("my_test.txt","outt.txt")
```


# MeKG_tools
