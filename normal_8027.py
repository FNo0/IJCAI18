# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 16:03:52 2018

@author: FNo0
"""

import pandas as pd
import numpy as np
import math
import datetime
import pickle
import os
import Baysian
import warnings
warnings.filterwarnings("ignore")

def evaluate(answer,predict):
    '''
    answer:dataframe,包含instance_id和is_trade
    predict:dataframe,包含instance_id和predicted_score
    公式:logloss = -1/N * Σ(yi*log(pi)+(1-yi)*log(1-pi))
    越小越好
    '''
    N = len(answer)
    result = pd.merge(answer,predict,how = 'left',on = 'instance_id') # dataframe,包含instance_id,is_trade,predicted_score
    logloss = list(map(lambda y,p : y * math.log(p) + (1 - y) * math.log(1 - p),result['is_trade'],result['predicted_score'])) # yi*log(pi)+(1-yi)*log(1-pi)
    logloss = pd.DataFrame(logloss)
    logloss = - (1 / N) * sum(logloss[0]) # logloss = -1/N * Σ(yi*log(pi)+(1-yi)*log(1-pi))
    return logloss

def split_vali(source_train):
    va_label_date = datetime.datetime(2018,9,24)
    va_feat_start_date = va_label_date - datetime.timedelta(days = 2)
    va_label = source_train[source_train['time'].map(lambda x : datetime.datetime(x.year,x.month,x.day) == va_label_date)] # 验证集标签
    va_feat = source_train[(source_train['time'] >= va_feat_start_date) & (source_train['time'] < va_label_date)]
    return va_feat,va_label

def split_train(source_train,label_day):
    tr_label_date = datetime.datetime(2018,9,label_day)
    tr_feat_start_date = tr_label_date - datetime.timedelta(days = 2)
    tr_label = source_train[source_train['time'].map(lambda x : datetime.datetime(x.year,x.month,x.day) == tr_label_date)] # 验证集标签
    tr_feat = source_train[(source_train['time'] >= tr_feat_start_date) & (source_train['time'] < tr_label_date)]
    return tr_feat,tr_label

def get_l_rank_feature(dataset):
    '''
    排序特征
    '''
    data = dataset.copy()
    
    # item被点击升序排名
    up = data.groupby(['item_id'])['time'].rank(ascending = True)
    data['l_item_id_click_rank_up'] = up
    # item被点击降序排名
    down = data.groupby(['item_id'])['time'].rank(ascending = False)
    data['l_item_id_click_rank_down'] = down
    
    # user被点击升序排名
    up = data.groupby(['user_id'])['time'].rank(ascending = True)
    data['l_user_id_click_rank_up'] = up
    # user被点击降序排名
    down = data.groupby(['user_id'])['time'].rank(ascending = False)
    data['l_user_id_click_rank_down'] = down
    
    # shop被点击升序排名
    up = data.groupby(['shop_id'])['time'].rank(ascending = True)
    data['l_shop_id_click_rank_up'] = up
    # shop被点击降序排名
    down = data.groupby(['shop_id'])['time'].rank(ascending = False)
    data['l_shop_id_click_rank_down'] = down
    
    # user-item被点击升序排名
    up = data.groupby(['user_id','item_id'])['time'].rank(ascending = True)
    data['l_user_id_item_id_click_rank_up'] = up
    # user-item被点击降序排名
    down = data.groupby(['user_id','item_id'])['time'].rank(ascending = False)
    data['l_user_id_item_id_click_rank_down'] = down
    
    # user-shop被点击升序排名
    up = data.groupby(['user_id','shop_id'])['time'].rank(ascending = True)
    data['l_user_id_shop_id_click_rank_up'] = up
    # user-shop被点击降序排名
    down = data.groupby(['user_id','shop_id'])['time'].rank(ascending = False)
    data['l_user_id_shop_id_click_rank_down'] = down

    # 删除不需要的
    data.drop(['user_id','item_id','shop_id','time'],axis = 1,inplace = True)
    
    # 返回
    return data

'''
特征部分特征
'''

def get_f_item_feature(dataset):
    '''
    item特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['item_id'])
    # 特征名前缀
    prefixs = 'f_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # item被点击数
    pivot = pd.pivot_table(data,index = keys,values = 'user_id',aggfunc = len)
    pivot.rename(columns = {'user_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # item成交数
    pivot = pd.pivot_table(data,index = keys,values = 'is_trade',aggfunc = sum)
    pivot.rename(columns = {'is_trade' : prefixs + 'trade_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature

def get_f_category_feature(dataset):
    '''
    category特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['item_category'])
    # 特征名前缀
    prefixs = 'f_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # category点击数
    pivot = pd.pivot_table(data,index = keys,values = 'user_id',aggfunc = len)
    pivot.rename(columns = {'user_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # category成交数
    pivot = pd.pivot_table(data,index = keys,values = 'is_trade',aggfunc = sum)
    pivot.rename(columns = {'is_trade' : prefixs + 'trade_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature

def get_f_user_feature(dataset):
    '''
    user特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['user_id'])
    # 特征名前缀
    prefixs = 'f_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # user点击数
    pivot = pd.pivot_table(data,index = keys,values = 'item_id',aggfunc = len)
    pivot.rename(columns = {'item_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # user成交数
    pivot = pd.pivot_table(data,index = keys,values = 'is_trade',aggfunc = sum)
    pivot.rename(columns = {'is_trade' : prefixs + 'trade_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature

def get_f_user_star_feature(dataset):
    '''
    user_star特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['user_star_level'])
    # 特征名前缀
    prefixs = 'f_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # user_star点击数
    pivot = pd.pivot_table(data,index = keys,values = 'item_id',aggfunc = len)
    pivot.rename(columns = {'item_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # user_star成交数
    pivot = pd.pivot_table(data,index = keys,values = 'is_trade',aggfunc = sum)
    pivot.rename(columns = {'is_trade' : prefixs + 'trade_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature

def get_f_shop_feature(dataset):
    '''
    shop特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['shop_id'])
    # 特征名前缀
    prefixs = 'f_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # shop被点击数
    pivot = pd.pivot_table(data,index = keys,values = 'user_id',aggfunc = len)
    pivot.rename(columns = {'user_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # shop成交数
    pivot = pd.pivot_table(data,index = keys,values = 'is_trade',aggfunc = sum)
    pivot.rename(columns = {'is_trade' : prefixs + 'trade_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature

def get_f_user_item_feature(dataset):
    '''
    user_item特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['user_id','item_id'])
    # 特征名前缀
    prefixs = 'f_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # user-item点击数
    pivot = pd.pivot_table(data,index = keys,values = 'shop_id',aggfunc = len)
    pivot.rename(columns = {'shop_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # user-item成交数
    pivot = pd.pivot_table(data,index = keys,values = 'is_trade',aggfunc = sum)
    pivot.rename(columns = {'is_trade' : prefixs + 'trade_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature

def get_f_user_category_feature(dataset):
    '''
    user_category特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['user_id','item_category'])
    # 特征名前缀
    prefixs = 'f_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # user-category点击数
    pivot = pd.pivot_table(data,index = keys,values = 'item_id',aggfunc = len)
    pivot.rename(columns = {'item_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # user-category成交数
    pivot = pd.pivot_table(data,index = keys,values = 'is_trade',aggfunc = sum)
    pivot.rename(columns = {'is_trade' : prefixs + 'trade_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature

def get_f_user_shop_feature(dataset):
    '''
    user_shop特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['user_id','shop_id'])
    # 特征名前缀
    prefixs = 'f_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # user-shop点击数
    pivot = pd.pivot_table(data,index = keys,values = 'item_id',aggfunc = len)
    pivot.rename(columns = {'item_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # user-shop成交数
    pivot = pd.pivot_table(data,index = keys,values = 'is_trade',aggfunc = sum)
    pivot.rename(columns = {'is_trade' : prefixs + 'trade_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature

def get_f_item_hour_feature(dataset):
    '''
    item-hour特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['item_id','hour'])
    # 特征名前缀
    prefixs = 'f_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # item-hour点击数
    pivot = pd.pivot_table(data,index = keys,values = 'user_id',aggfunc = len)
    pivot.rename(columns = {'user_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # item-hour成交数
    pivot = pd.pivot_table(data,index = keys,values = 'is_trade',aggfunc = sum)
    pivot.rename(columns = {'is_trade' : prefixs + 'trade_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature 

def get_f_category_hour_feature(dataset):
    '''
    category-hour特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['item_category','hour'])
    # 特征名前缀
    prefixs = 'f_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # category-hour点击数
    pivot = pd.pivot_table(data,index = keys,values = 'user_id',aggfunc = len)
    pivot.rename(columns = {'user_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # category-hour成交数
    pivot = pd.pivot_table(data,index = keys,values = 'is_trade',aggfunc = sum)
    pivot.rename(columns = {'is_trade' : prefixs + 'trade_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature

def get_f_user_hour_feature(dataset):
    '''
    user-hour特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['user_id','hour'])
    # 特征名前缀
    prefixs = 'f_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # user-hour点击数
    pivot = pd.pivot_table(data,index = keys,values = 'item_id',aggfunc = len)
    pivot.rename(columns = {'item_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # user-hour成交数
    pivot = pd.pivot_table(data,index = keys,values = 'is_trade',aggfunc = sum)
    pivot.rename(columns = {'is_trade' : prefixs + 'trade_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature 

def get_f_shop_hour_feature(dataset):
    '''
    shop-hour特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['shop_id','hour'])
    # 特征名前缀
    prefixs = 'f_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # shop-hour点击数
    pivot = pd.pivot_table(data,index = keys,values = 'user_id',aggfunc = len)
    pivot.rename(columns = {'user_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # shop-hour成交数
    pivot = pd.pivot_table(data,index = keys,values = 'is_trade',aggfunc = sum)
    pivot.rename(columns = {'is_trade' : prefixs + 'trade_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature 

def get_f_user_item_hour_feature(dataset):
    '''
    user-item-hour特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['user_id','item_id','hour'])
    # 特征名前缀
    prefixs = 'f_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # user-item-hour点击数
    pivot = pd.pivot_table(data,index = keys,values = 'shop_id',aggfunc = len)
    pivot.rename(columns = {'shop_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # user-item-hour成交数
    pivot = pd.pivot_table(data,index = keys,values = 'is_trade',aggfunc = sum)
    pivot.rename(columns = {'is_trade' : prefixs + 'trade_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature 

def get_f_user_category_hour_feature(dataset):
    '''
    user_category_hour特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['user_id','item_category','hour'])
    # 特征名前缀
    prefixs = 'f_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # user-category-hour点击数
    pivot = pd.pivot_table(data,index = keys,values = 'item_id',aggfunc = len)
    pivot.rename(columns = {'item_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # user-category-hour成交数
    pivot = pd.pivot_table(data,index = keys,values = 'is_trade',aggfunc = sum)
    pivot.rename(columns = {'is_trade' : prefixs + 'trade_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature

def get_f_user_shop_hour_feature(dataset):
    '''
    user-shop-hour特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['user_id','shop_id','hour'])
    # 特征名前缀
    prefixs = 'f_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # user-shop-hour点击数
    pivot = pd.pivot_table(data,index = keys,values = 'item_id',aggfunc = len)
    pivot.rename(columns = {'item_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # user-shop-hour成交数
    pivot = pd.pivot_table(data,index = keys,values = 'is_trade',aggfunc = sum)
    pivot.rename(columns = {'is_trade' : prefixs + 'trade_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature 

def get_f_item_pcaRank_feature(dataset):
    '''
    item_pcaRank特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['item_id','predict_category_rank'])
    # 特征名前缀
    prefixs = 'f_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # item-pcaRank点击数
    pivot = pd.pivot_table(data,index = keys,values = 'user_id',aggfunc = len)
    pivot.rename(columns = {'user_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # item-pcaRank成交数
    pivot = pd.pivot_table(data,index = keys,values = 'is_trade',aggfunc = sum)
    pivot.rename(columns = {'is_trade' : prefixs + 'trade_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature 

def get_f_user_pcaRank_feature(dataset):
    '''
    user_pcaRank特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['user_id','predict_category_rank'])
    # 特征名前缀
    prefixs = 'f_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # user-pcaRank点击数
    pivot = pd.pivot_table(data,index = keys,values = 'item_id',aggfunc = len)
    pivot.rename(columns = {'item_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # user-pcaRank成交数
    pivot = pd.pivot_table(data,index = keys,values = 'is_trade',aggfunc = sum)
    pivot.rename(columns = {'is_trade' : prefixs + 'trade_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature 

def get_f_shop_pcaRank_feature(dataset):
    '''
    shop_pcaRank特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['shop_id','predict_category_rank'])
    # 特征名前缀
    prefixs = 'f_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # shop-pcaRank点击数
    pivot = pd.pivot_table(data,index = keys,values = 'user_id',aggfunc = len)
    pivot.rename(columns = {'user_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # shop-pcaRank成交数
    pivot = pd.pivot_table(data,index = keys,values = 'is_trade',aggfunc = sum)
    pivot.rename(columns = {'is_trade' : prefixs + 'trade_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature 

def get_f_user_item_pcaRank_feature(dataset):
    '''
    user_item_pcaRank特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['user_id','item_id','predict_category_rank'])
    # 特征名前缀
    prefixs = 'f_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # user-item-pcaRank点击数
    pivot = pd.pivot_table(data,index = keys,values = 'shop_id',aggfunc = len)
    pivot.rename(columns = {'shop_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # user-item-pcaRank成交数
    pivot = pd.pivot_table(data,index = keys,values = 'is_trade',aggfunc = sum)
    pivot.rename(columns = {'is_trade' : prefixs + 'trade_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature 

def get_f_user_shop_pcaRank_feature(dataset):
    '''
    user_shop_pcaRank特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['user_id','shop_id','predict_category_rank'])
    # 特征名前缀
    prefixs = 'f_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # user-shop-pcaRank点击数
    pivot = pd.pivot_table(data,index = keys,values = 'item_id',aggfunc = len)
    pivot.rename(columns = {'item_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # user-shop-pcaRank成交数
    pivot = pd.pivot_table(data,index = keys,values = 'is_trade',aggfunc = sum)
    pivot.rename(columns = {'is_trade' : prefixs + 'trade_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature 

'''
标签部分特征
'''

def get_l_item_feature(dataset):
    '''
    item特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['item_id'])
    # 特征名前缀
    prefixs = 'l_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # item被点击数
    pivot = pd.pivot_table(data,index = keys,values = 'user_id',aggfunc = len)
    pivot.rename(columns = {'user_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer') 
    # 
    pass
    # 返回
    return feature

def get_l_user_feature(dataset):
    '''
    user特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['user_id'])
    # 特征名前缀
    prefixs = 'l_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # user点击数
    pivot = pd.pivot_table(data,index = keys,values = 'item_id',aggfunc = len)
    pivot.rename(columns = {'item_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature

def get_l_shop_feature(dataset):
    '''
    shop特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['shop_id'])
    # 特征名前缀
    prefixs = 'l_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # shop被点击数
    pivot = pd.pivot_table(data,index = keys,values = 'user_id',aggfunc = len)
    pivot.rename(columns = {'user_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature

def get_l_user_item_feature(dataset):
    '''
    user_item特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['user_id','item_id'])
    # 特征名前缀
    prefixs = 'l_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # user-item点击数
    pivot = pd.pivot_table(data,index = keys,values = 'shop_id',aggfunc = len)
    pivot.rename(columns = {'shop_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer') 
    # 
    pass
    # 返回
    return feature

def get_l_user_shop_feature(dataset):
    '''
    user_shop特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['user_id','shop_id'])
    # 特征名前缀
    prefixs = 'l_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # user-shop点击数
    pivot = pd.pivot_table(data,index = keys,values = 'item_id',aggfunc = len)
    pivot.rename(columns = {'item_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer') 
    # 
    pass
    # 返回
    return feature

def get_l_item_hour_feature(dataset):
    '''
    item_hour特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['item_id','hour'])
    # 特征名前缀
    prefixs = 'l_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # item-hour被点击数
    pivot = pd.pivot_table(data,index = keys,values = 'user_id',aggfunc = len)
    pivot.rename(columns = {'user_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer') 
    # 
    pass
    # 返回
    return feature

def get_l_user_hour_feature(dataset):
    '''
    user_hour特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['user_id','hour'])
    # 特征名前缀
    prefixs = 'l_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # user-hour点击数
    pivot = pd.pivot_table(data,index = keys,values = 'item_id',aggfunc = len)
    pivot.rename(columns = {'item_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature

def get_l_shop_hour_feature(dataset):
    '''
    shop_hour特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['shop_id','hour'])
    # 特征名前缀
    prefixs = 'l_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # shop-hour被点击数
    pivot = pd.pivot_table(data,index = keys,values = 'user_id',aggfunc = len)
    pivot.rename(columns = {'user_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer')
    # 
    pass
    # 返回
    return feature

def get_l_user_item_hour_feature(dataset):
    '''
    user_item_hour特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['user_id','item_id','hour'])
    # 特征名前缀
    prefixs = 'l_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # user-item-hour点击数
    pivot = pd.pivot_table(data,index = keys,values = 'shop_id',aggfunc = len)
    pivot.rename(columns = {'shop_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer') 
    # 
    pass
    # 返回
    return feature

def get_l_user_shop_hour_feature(dataset):
    '''
    user_shop_hour特征
    '''
    data = dataset.copy()
    # 主键
    keys = list(['user_id','shop_id','hour'])
    # 特征名前缀
    prefixs = 'l_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 返回的特征
    feature = pd.DataFrame(columns = keys)
    # user-shop-hour点击数
    pivot = pd.pivot_table(data,index = keys,values = 'item_id',aggfunc = len)
    pivot.rename(columns = {'item_id' : prefixs + 'click_cnt'},inplace = True)
    pivot.reset_index(inplace = True)
    feature = pd.merge(feature,pivot,on = keys,how = 'outer') 
    # 
    pass
    # 返回
    return feature

def create_dataset(feat_part,label_part,flag):
    # 原数据
    dataset_feat = feat_part.copy()
    dataset = label_part.copy()
    print('    提取特征:')
    ## 特征部分特征
    print('        f部分特征:')
    # f_item特征
    f_item_feature = get_f_item_feature(dataset_feat)
    dataset = pd.merge(dataset,f_item_feature,on = 'item_id',how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            f_item特征提取完毕!')
    # f_category特征
    f_category_feature = get_f_category_feature(dataset_feat)
    dataset = pd.merge(dataset,f_category_feature,on = 'item_category',how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            f_category特征提取完毕!')
    # f_user特征
    f_user_feature = get_f_user_feature(dataset_feat)
    dataset = pd.merge(dataset,f_user_feature,on = 'user_id',how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            f_user特征提取完毕!')
    # f_user_star特征
    f_user_star_feature = get_f_user_star_feature(dataset_feat)
    dataset = pd.merge(dataset,f_user_star_feature,on = 'user_star_level',how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            f_user_star特征提取完毕!')
    # f_shop特征
    f_shop_feature = get_f_shop_feature(dataset_feat)
    dataset = pd.merge(dataset,f_shop_feature,on = 'shop_id',how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            f_shop特征提取完毕!')
    # f_user-item特征
    f_user_item_feature = get_f_user_item_feature(dataset_feat)
    dataset = pd.merge(dataset,f_user_item_feature,on = ['user_id','item_id'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            f_user-item特征提取完毕!')
    # f_user-category特征
    f_user_category_feature = get_f_user_category_feature(dataset_feat)
    dataset = pd.merge(dataset,f_user_category_feature,on = ['user_id','item_category'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            f_user-category特征提取完毕!')
    # f_user-shop特征
    f_user_shop_feature = get_f_user_shop_feature(dataset_feat)
    dataset = pd.merge(dataset,f_user_shop_feature,on = ['user_id','shop_id'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            f_user-shop特征提取完毕!')
    # f_item-hour特征
    f_item_hour_feature = get_f_item_hour_feature(dataset_feat)
    dataset = pd.merge(dataset,f_item_hour_feature,on = ['item_id','hour'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            f_item-hour特征提取完毕!')
    # f_category-hour特征
    f_category_hour_feature = get_f_category_hour_feature(dataset_feat)
    dataset = pd.merge(dataset,f_category_hour_feature,on = ['item_category','hour'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            f_category-hour特征提取完毕!')
    # f_user-hour特征
    f_user_hour_feature = get_f_user_hour_feature(dataset_feat)
    dataset = pd.merge(dataset,f_user_hour_feature,on = ['user_id','hour'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            f_user-hour特征提取完毕!')
    # f_shop-hour特征
    f_shop_hour_feature = get_f_shop_hour_feature(dataset_feat)
    dataset = pd.merge(dataset,f_shop_hour_feature,on = ['shop_id','hour'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            f_shop-hour特征提取完毕!')
    # f_user-item-hour特征
    f_user_item_hour_feature = get_f_user_item_hour_feature(dataset_feat)
    dataset = pd.merge(dataset,f_user_item_hour_feature,on = ['user_id','item_id','hour'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            f_user-item-hour特征提取完毕!')
    # f_user-category-hour特征
    f_user_category_hour_feature = get_f_user_category_hour_feature(dataset_feat)
    dataset = pd.merge(dataset,f_user_category_hour_feature,on = ['user_id','item_category','hour'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            f_user-category-hour特征提取完毕!')
    # f_user-shop-hour特征
    f_user_shop_hour_feature = get_f_user_shop_hour_feature(dataset_feat)
    dataset = pd.merge(dataset,f_user_shop_hour_feature,on = ['user_id','shop_id','hour'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            f_user-shop-hour特征提取完毕!')
    # f_item-pcaRank特征
    f_item_pcaRank_feature = get_f_item_pcaRank_feature(dataset_feat)
    dataset = pd.merge(dataset,f_item_pcaRank_feature,on = ['item_id','predict_category_rank'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            f_item-pcaRank特征提取完毕!')
    # f_user-pcaRank特征
    f_user_pcaRank_feature = get_f_user_pcaRank_feature(dataset_feat)
    dataset = pd.merge(dataset,f_user_pcaRank_feature,on = ['user_id','predict_category_rank'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            f_user-pcaRank特征提取完毕!')
    # f_shop-pcaRank特征
    f_shop_pcaRank_feature = get_f_shop_pcaRank_feature(dataset_feat)
    dataset = pd.merge(dataset,f_shop_pcaRank_feature,on = ['shop_id','predict_category_rank'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            f_shop-pcaRank特征提取完毕!')
    # f_user-item-pcaRank特征
    f_user_item_pcaRank_feature = get_f_user_item_pcaRank_feature(dataset_feat)
    dataset = pd.merge(dataset,f_user_item_pcaRank_feature,on = ['user_id','item_id','predict_category_rank'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            f_user-item-pcaRank特征提取完毕!')
    # f_user-shop-pcaRank特征
    f_user_shop_pcaRank_feature = get_f_user_shop_pcaRank_feature(dataset_feat)
    dataset = pd.merge(dataset,f_user_shop_pcaRank_feature,on = ['user_id','shop_id','predict_category_rank'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            f_user-shop-pcaRank特征提取完毕!')
    
    # 
    pass
    print('        f部分特征提取完毕!')
    ## 标签部分特征
    print('        l部分特征:')
    # l_item特征
    l_item_feature = get_l_item_feature(dataset)
    dataset = pd.merge(dataset,l_item_feature,on = 'item_id',how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            l_item特征提取完毕!')
    # l_user特征
    l_user_feature = get_l_user_feature(dataset)
    dataset = pd.merge(dataset,l_user_feature,on = 'user_id',how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            l_user特征提取完毕!')
    # l_shop特征
    l_shop_feature = get_l_shop_feature(dataset)
    dataset = pd.merge(dataset,l_shop_feature,on = 'shop_id',how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            l_shop特征提取完毕!')
    # l_user-item特征
    l_user_item_feature = get_l_user_item_feature(dataset)
    dataset = pd.merge(dataset,l_user_item_feature,on = ['user_id','item_id'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            l_user-item特征提取完毕!')
    # l_user-shop特征
    l_user_shop_feature = get_l_user_shop_feature(dataset)
    dataset = pd.merge(dataset,l_user_shop_feature,on = ['user_id','shop_id'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            l_user-shop特征提取完毕!')
    # l_item-hour特征
    l_item_hour_feature = get_l_item_hour_feature(dataset)
    dataset = pd.merge(dataset,l_item_hour_feature,on = ['item_id','hour'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            l_item-hour特征提取完毕!')
    # l_user-hour特征
    l_user_hour_feature = get_l_user_hour_feature(dataset)
    dataset = pd.merge(dataset,l_user_hour_feature,on = ['user_id','hour'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            l_user-hour特征提取完毕!')
    # l_shop-hour特征
    l_shop_hour_feature = get_l_shop_hour_feature(dataset)
    dataset = pd.merge(dataset,l_shop_hour_feature,on = ['shop_id','hour'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            l_shop-hour特征提取完毕!')
    # l_user-item-hour特征
    l_user_item_hour_feature = get_l_user_item_hour_feature(dataset)
    dataset = pd.merge(dataset,l_user_item_hour_feature,on = ['user_id','item_id','hour'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            l_user-item-hour特征提取完毕!')
    # l_user-shop-hour特征
    l_user_shop_hour_feature = get_l_user_shop_hour_feature(dataset)
    dataset = pd.merge(dataset,l_user_shop_hour_feature,on = ['user_id','shop_id','hour'],how = 'left')
    dataset.fillna(0,downcast = 'infer',inplace = True)
    print('            l_user-shop-hour特征提取完毕!')
    # 复合特征
    dataset['l_user_id_item_id_click_div_user_id_click'] = dataset['l_user_id_item_id_click_cnt'] / dataset['l_user_id_click_cnt']
    print('            l_复合特征提取完毕!')
    # rank特征
    label_date = np.max(dataset['time'])
    label_date = label_date.day
    dump_path = r'cache/' + str(label_date) + '_rank'
    if os.path.exists(dump_path):
        print('                直接读取rank特征...')
        rank = pickle.load(open(dump_path,'rb'))
    else:
        print('                构造rank特征...')
        sub_dataset = dataset[['user_id','item_id','shop_id','time']]
        rank = get_l_rank_feature(sub_dataset)
        pickle.dump(rank,open(dump_path,'wb'))
    dataset = pd.concat([dataset,rank],axis = 1)
    print('            l_rank特征提取完毕!')
    # 
    pass
    print('        l部分特征提取完毕!')
    # 删不需要的
    dels = list(['item_category_list','item_property_list','item_brand_id',
                 'context_id','context_timestamp','predict_category_property',
                 'predict_category'])
    dataset.drop(dels,axis = 1,inplace = True)
    # 贝叶斯平滑rate
    dataset = get_f_rate(dataset,list(['item_id']))
    dataset = get_f_rate(dataset,list(['item_category']))
    dataset = get_f_rate(dataset,list(['user_id']))
    dataset = get_f_rate(dataset,list(['user_star_level']))
    dataset = get_f_rate(dataset,list(['shop_id']))
    dataset = get_f_rate(dataset,list(['user_id','item_id']))
    dataset = get_f_rate(dataset,list(['user_id','item_category']))
    dataset = get_f_rate(dataset,list(['user_id','shop_id']))
    dataset = get_f_rate(dataset,list(['item_id','hour']))
    dataset = get_f_rate(dataset,list(['item_category','hour']))
    dataset = get_f_rate(dataset,list(['user_id','hour']))
    dataset = get_f_rate(dataset,list(['shop_id','hour']))
    dataset = get_f_rate(dataset,list(['user_id','item_id','hour']))
    dataset = get_f_rate(dataset,list(['user_id','item_category','hour']))
    dataset = get_f_rate(dataset,list(['user_id','shop_id','hour']))
    dataset.fillna(0,downcast = 'infer',inplace = True)
    # 删多余id
    dataset.drop(['item_id','item_category','user_id','shop_id','hour','time'],axis = 1,inplace = True)
    # 特征提完
    print('    特征提取完毕!')
    # 返回
    return dataset

def get_f_rate(dataset,keys):
    '''
    f部分的转换率特征
    '''
    data = dataset.copy()
    # 前缀
    prefixs = 'f_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # bayes平滑
    date = np.max(data['time'])
    date = date.day
    dump_path = r'cache2/' + prefixs + '_rate_' + str(date)
    print('        ' + dump_path.split('/')[1] + '转换率:')
    if os.path.exists(dump_path):
        print('            直接读取贝叶斯平滑rate...')
        rate = pickle.load(open(dump_path,'rb'))
    else:
        print('            构造贝叶斯平滑rate...')
        bs = Baysian.BayesianSmoothing(1, 1)
        bs.update(data[prefixs + 'click_cnt'].values, data[prefixs + 'trade_cnt'].values, 1000, 0.001)
        rate = (data[prefixs + 'trade_cnt'] + bs.alpha) / (data[prefixs + 'click_cnt'] + bs.alpha + bs.beta)
        pickle.dump(rate,open(dump_path,'wb'))
    data[prefixs + 'rate'] = rate
    # 返回
    return data
    
if __name__ == '__main__':
    # 原始数据
    source_train = pd.read_table('../data/round1_ijcai_18_train_20180301/round1_ijcai_18_train_20180301.txt',delim_whitespace = True)
    source_test = pd.read_table('../data/round1_ijcai_18_test_a_20180301/round1_ijcai_18_test_a_20180301.txt',delim_whitespace = True)

    # UNIX时间戳转换为北京时间
    source_train['time'] = list(source_train['context_timestamp'].map(lambda x : datetime.datetime.utcfromtimestamp(x) + datetime.timedelta(hours = 8))) # 东八区
    source_test['time'] = list(source_test['context_timestamp'].map(lambda x : datetime.datetime.utcfromtimestamp(x) + datetime.timedelta(hours = 8))) # 东八区,都是2018-09-25  
    source_train['hour'] = source_train['time'].map(lambda x : x.hour)
    source_test['hour'] = source_test['time'].map(lambda x : x.hour)
    
    ## 处理item-category_list
    # source_train
    item_category_train = pd.DataFrame(source_train['item_category_list'].map(lambda x : x.split(';')))
    item_category_train.rename(columns = {'item_category_list' : 'item_category'},inplace = True)
    item_category_train['item_category'] = item_category_train['item_category'].map(lambda x : x[2] if len(x) > 2 else x[1])
    source_train = pd.concat([source_train,item_category_train],axis = 1)
    # source_test
    item_category_test = pd.DataFrame(source_test['item_category_list'].map(lambda x : x.split(';')))
    item_category_test.rename(columns = {'item_category_list' : 'item_category'},inplace = True)
    item_category_test['item_category'] = item_category_test['item_category'].map(lambda x : x[2] if len(x) > 2 else x[1])
    source_test = pd.concat([source_test,item_category_test],axis = 1)
    
    ## 处理predict_category_property
    # source_train
    source_train['predict_category'] = source_train['predict_category_property'].map(lambda x : [y.split(':')[0] for y in x.split(';')])
    source_train['predict_category_rank'] = list(map(lambda x,y : y.index(x) if x in y else -1,source_train['item_category'],source_train['predict_category']))
    # source_test
    source_test['predict_category'] = source_test['predict_category_property'].map(lambda x : [y.split(':')[0] for y in x.split(';')])
    source_test['predict_category_rank'] = list(map(lambda x,y : y.index(x) if x in y else -1,source_test['item_category'],source_test['predict_category']))
    
    '''
    线下
    '''
    ## 训练集
    # 划分线下训练集
    tr1_feat,tr1_label = split_train(source_train,23)
    tr2_feat,tr2_label = split_train(source_train,22)
    tr3_feat,tr3_label = split_train(source_train,21)
    tr4_feat,tr4_label = split_train(source_train,20)
    # 构造线下训练集
    print('构造tr1:')
    tr1 = create_dataset(tr1_feat,tr1_label,'offline')
    print('tr1构造完成!\n')
    print('构造tr2:')
    tr2 = create_dataset(tr2_feat,tr2_label,'offline')
    print('tr2构造完成!\n')
    print('构造tr3:')
    tr3 = create_dataset(tr3_feat,tr3_label,'offline')
    print('tr3构造完成!\n')
    print('构造tr4:')
    tr4 = create_dataset(tr4_feat,tr4_label,'offline')
    print('tr4构造完成!\n')
    tr_off = pd.concat([tr1,tr2,tr3,tr4],axis = 0)

    ## 验证集
    # 划分线下验证集
    va_feat,va_label = split_vali(source_train)
    # 构造线下验证集
    print('构造va:')
    va = create_dataset(va_feat,va_label,'offline')
    print('va构造完成!\n')
    
    ## 保存线下训练集验证集
    tr_off.to_csv('offline/train.csv',index = False)
    va.to_csv('offline/vali.csv',index = False)
    
    
    '''
    线上
    '''
    ## 训练集
    # 划分线上训练集
    tr5_feat,tr5_label = split_train(source_train,24)
    # 构造线上训练集
    print('tr1~tr4已构造完全!')
    print('构造tr5:')
    tr5 = create_dataset(tr5_feat,tr5_label,'online')
    print('tr5构造完成!\n')
    tr_on = pd.concat([tr5,tr_off],axis = 0)
    
    ## 测试集
    te_feat = source_train[source_train['time'].map(lambda x : (x.day == 24) or (x.day ==  23))]
    te_label = source_test.copy()
    # 构造线下验证集
    print('构造te:')
    te = create_dataset(te_feat,te_label,'online')
    print('te构造完成!\n')
    
    ## 保存
    tr_on.to_csv('online/train.csv',index = False)
    te.to_csv('online/test.csv',index = False)
    