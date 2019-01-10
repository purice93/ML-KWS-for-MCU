""" 
@author: zoutai
@file: testReshape.py 
@time: 2018/12/19 
@description: 测试reshape的逻辑，是以行为单位
一个坑：转置权重
"""
import numpy as np
origin=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
update = np.reshape(origin,[3,4])
update2 = np.reshape(origin,[4,3])
update3 = np.reshape(origin,[2,2,3])
print(update,update2,update3,sep='\n')