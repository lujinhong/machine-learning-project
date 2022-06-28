# -*- coding: utf-8 -*- 

"""
    AUTHOR: lujinhong
CREATED ON: 2021年11月18日 17:23
   PROJECT: machine-learning-project
   DESCRIPTION: 项目中经常用到的常量。
"""
import os

dataset_root = os.path.join(os.environ['HOME'], 'datasets')
model_root = os.path.join(os.environ['HOME'], 'models')


# 常用数据集的下载url
DATASET_URL = dict()
DATASET_URL['time_machine'] = 'https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt'
DATASET_URL['machine_translation_fra-eng'] = 'http://www.manythings.org/anki/fra-eng.zip'
DATASET_URL['wikitext-2'] = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'



