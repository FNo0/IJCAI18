# -*- coding: utf-8 -*-
"""
Created on Tue Jul 4 10:15:17 2017

@author: yuwei
"""

import pandas as pd
import xgboost as xgb
import math

def model_xgb(file_train,file_vali):
    train = pd.read_csv(file_train)
    print('训练集读取完成!')
    vali = pd.read_csv(file_vali)
    print('验证集读取完成!')
    
    train_y = train['is_trade'].values
    train_x = train.drop(['instance_id','is_trade'],axis=1).values
    vali_x = vali.drop(['instance_id','is_trade'],axis=1).values
 
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvali = xgb.DMatrix(vali_x)
    
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
    predict = bst.predict(dvali)
    vali_xy = vali[['instance_id']]
    vali_xy['predicted_score'] = predict
    return vali_xy

def evaluate1(answer,predict):
    '''
    answer:dataframe,包含instance_id和is_trade
    predict:dataframe,包含instance_id和predicted_score
    公式:logloss = -1/N * Σ(yi*log(pi)+(1-yi)*log(1-pi))
    越小越好
    '''
    N = len(answer)
    result = pd.concat([answer,predict[['predicted_score']]],axis = 1) # dataframe,包含instance_id,is_trade,predicted_score
    logloss = list(map(lambda y,p : y * math.log(p) + (1 - y) * math.log(1 - p),result['is_trade'],result['predicted_score'])) # yi*log(pi)+(1-yi)*log(1-pi)
    logloss = pd.DataFrame(logloss)
    logloss = - (1 / N) * sum(logloss[0]) # logloss = -1/N * Σ(yi*log(pi)+(1-yi)*log(1-pi))
    return logloss

import scipy as sp
def evaluate2(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll
    
    
if __name__ == '__main__':
    file_train = r'offline/train.csv'
    file_vali = r'offline/vali.csv'
    predict = model_xgb(file_train,file_vali)
        
    answer = pd.read_csv(file_vali)
    answer = answer[['instance_id','is_trade']]
    logloss1 = evaluate1(answer,predict)
        
    act = answer['is_trade'].tolist()
    pred = predict['predicted_score'].tolist()
    logloss2 = evaluate2(act,pred)
        
        


        

        

        
        