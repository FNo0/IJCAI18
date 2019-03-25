# -*- coding: utf-8 -*-
"""
Created on Tue Jul 4 10:15:17 2017

@author: yuwei
"""

import pandas as pd
import xgboost as xgb

def model_xgb(file_train,file_test):
    train = pd.read_csv(file_train)
    print('训练集读取完成!')
    test = pd.read_csv(file_test)
    print('验证集读取完成!')
    
    train_y = train['is_trade'].values
    train_x = train.drop(['instance_id','is_trade'],axis=1).values
    test_x = test.drop(['instance_id'],axis=1).values
 
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)
    
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'binary:logistic',
              'eval_metric' : 'logloss',
              'eta': 0.03,
              'max_depth': 6,  # 4 3
              'colsample_bytree': 0.8,#0.8
              'subsample': 0.8,
              'scale_pos_weight': 1,
              'min_child_weight': 18  # 2 3
              }
    # 训练
    print('开始训练!')
    watchlist = [(dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_boost_round=600,evals=watchlist)
    # 预测
    print('开始预测!')
    predict = bst.predict(dtest)
    test_xy = test[['instance_id']]
    test_xy['predicted_score'] = predict
    return test_xy

    
if __name__ == '__main__':
    file_train = r'online/train.csv'
    file_test = r'online/test.csv'
    predict = model_xgb(file_train,file_test)
    
    predict.to_csv('result/normal_8027.txt',index = False,sep = ' ',line_terminator = '\r')



        

        
        