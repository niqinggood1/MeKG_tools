#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：CMeKG_tools-main 
@File    ：test.py
@IDE     ：PyCharm 
@Author  ：patrick
@Date    ：2023/5/1 18:07 
'''
import numpy as np
if __name__ == '__main__':
    a   =  np.array([2, 4, 6, 8, 10])
    # 只有一个参数表示条件的时候
    dd  =  np.where(a > 5)
    print('dd',dd)
    subjects    = [(1,3),(5,10),(12,15)]
    import  torch
    subject_ids = torch.tensor(subjects).view(1, -1)
    print(  'subject_ids:', subject_ids )
    exit()
    
  
  