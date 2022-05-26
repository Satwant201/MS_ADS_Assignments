#!/usr/bin/env python
# coding: utf-8

# In[1]:




'''
Method Description:
Model Type - Content Based Predictive regression model.

Initially I was building  a hybrid model using Item based and content based base learners.. But my overall rmse was same as content based base model.

Then I decided to improve upon my predifcitve model.

I did extensive feature engineering and curated around 14 features using diffrent datasets named review, tips, business,user json files.

then I performed hyper parameter tuing on learnign rate, depth and number of trees. 
Error Distribution:
>=0 & <1:	102017
>=1 & <2:	35870
>=2 & <3:	13753
>=3 & <4:	2352
>=4 & <5:	4

RMSE:
0.979880746

Execution Time:
754.065 seconds

'''

##################################################Library Imports #########################################################################


import numpy as np
import json

import os
from collections import defaultdict

import operator
import sys
import time


from pyspark import SparkContext

sc = SparkContext()#conf=conf)
sc.setLogLevel("ERROR")

#################################################################################################################################################

##############################################Function Defintions ###############################################################################


def extract_nested_feature(dictionary,paramter):
    if dictionary is None:
        return 0
    target = paramter
    val= dictionary.get(target,0)
    return float(val)

def func(x):
    return business_photo_cnt.get(x,0)

def func2(x):
    cnt_dict = business_neigh_fict.get(x,None)
    if cnt_dict is None:
        return mean_neighbours
    return cnt_dict['user_id']


def func3(x):
    cnt_dict = users_neigh_fict.get(x,None)
    if cnt_dict is None:
        return mean_neighbours_user
    return cnt_dict['business_id']


def func4(x):
    
    return business_postal_avg.get(x,0)

#################################################################################################################################################


start_time = time.time()



print("Loading necessary system arguments")


# train_folder = "../Data/"
# test_file = "../Data/yelp_test_ans.csv"
# output_file = "../Data/taskoutput.txt"

train_folder = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]



tips_data = sc.textFile(os.path.join(train_folder, 'tip.json')).map(json.loads)



tips_base_data = tips_data.map(lambda x : ((x['business_id']),x['likes']))

countsByKey = sc.broadcast(tips_base_data.countByKey()) # SAMPLE OUTPUT of countsByKey.value: {u'2013-09-09': 215, u'2013-09-08': 69, ... snip ...}
rdd1 = tips_base_data.reduceByKey(operator.add) # Calculate the numerators (i.e. the SUMs).
rdd1 = rdd1.map(lambda x: (x[0], x[1]/countsByKey.value[x[0]])) # Divide ea
# tips_data_dict = .reduceByKey(operator.add).collectAsMap()


tips_data_dict1 =rdd1.collectAsMap()

tips_data_dict1_1 = defaultdict()

for key1,val1 in tips_data_dict1.items():
    tips_data_dict1_1[key1] = (val1-0)/(15)
    




business_rdd_read = sc.textFile(os.path.join(train_folder, 'business.json')).map(json.loads)

business_subset_rdd = business_rdd_read.map(lambda x : ((x['business_id']),(x['stars'],x['review_count'],extract_nested_feature(x['attributes'],"RestaurantsPriceRange2"))))

business_lookup = business_subset_rdd.collectAsMap()

avg_businesses_rating = business_subset_rdd.map(lambda x: x[1][0]).mean()
avg_businesses_reviewcount = business_subset_rdd.map(lambda x: x[1][1]).mean()


user_rdd_read = sc.textFile(os.path.join(train_folder, 'user.json')).map(json.loads)



user_rdd_features_rdd = user_rdd_read.map(lambda x : ((x["user_id"]),(x["average_stars"],x["review_count"],x['useful'],x['fans'])))

user_lookup = user_rdd_features_rdd.collectAsMap()

avg_users_rating = user_rdd_features_rdd.map(lambda x: x[1][0]).mean()
avg_users_reviewcount = user_rdd_features_rdd.map(lambda x: x[1][1]).mean()



traing_rdd_read = sc.textFile(train_folder+"yelp_train.csv")  ### added 

header = traing_rdd_read.take(1)[0] ### added 


training_rdd = traing_rdd_read.filter(lambda x : x!=header).map(lambda x : x.split(",")).map(lambda x : (x[0],x[1],float(x[2]))) ### addeed 



training_features = training_rdd.map(lambda x : ((x[0],x[1]),(user_lookup.get(x[0],(avg_users_reviewcount,avg_users_rating,0,0)),business_lookup.get(x[1],(avg_businesses_rating,avg_businesses_reviewcount,2)),tips_data_dict1_1.get(x[1],0),                                                              x[2])))




# train_features = get_features(train_data, user_json_map, business_json_map)
# train_features = fit_invalid_type_cols(train_features)

traininng_feature_formatted = training_features.map(lambda x : (x[1][0][0],
x[1][0][1],
x[1][0][2],
x[1][0][3],
x[1][1][0],
x[1][1][1],
x[1][1][2],
x[1][-2],                                                                
x[1][-1]))




# In[2]:


train_dataset_pd = traininng_feature_formatted.collect()


# In[3]:


import pandas as pd


# In[4]:



pd.DataFrame(train_dataset_pd,columns = ['x1','x2','x3','x4','x5','x6','x7','x8','target']).to_csv("Train_data_pd.csv",index=False)


# In[5]:



testing_rdd_read = sc.textFile(test_file)  ### added 

header = testing_rdd_read.take(1)[0] ### added 


test_rdd = testing_rdd_read.filter(lambda x : x!=header).map(lambda x : x.split(",")).map(lambda x : (x[0],x[1],2))

test_features = test_rdd.map(lambda x : ((x[0],x[1]),(user_lookup.get(x[0],(avg_users_reviewcount,avg_users_rating,0,0)),business_lookup.get(x[1],(avg_businesses_rating,avg_businesses_reviewcount,2)),tips_data_dict1_1.get(x[1],0),x[2])))


test_features_formatted = test_features.map(lambda x : ((x[0][0],x[0][1]),(x[1][0][0],
x[1][0][1],
x[1][0][2],
x[1][0][3],
x[1][1][0],
x[1][1][1],
x[1][1][2],
x[1][-2] ,                                                               
x[1][-1])))


# In[6]:


test_dataset_pd = test_features_formatted.collect()


# In[7]:


import pandas as pd


# In[8]:


pd.DataFrame([w[1] for w in test_dataset_pd],columns = ['x1','x2','x3','x4','x5','x6','x7','x8','target']).to_csv("val_dataset_pd.csv",index=False)


# In[9]:


import pandas as pd

import numpy as np

import json

yelp_train = pd.read_csv(train_folder+"yelp_train.csv")



yl_business_neighbours = yelp_train.groupby(['business_id']).count().reset_index()[['business_id','user_id']]

mean_neighbours = yl_business_neighbours['user_id'].mean()

business_neigh_fict =     yl_business_neighbours.set_index('business_id').T.to_dict()

yl_users_neighbours = yelp_train.groupby(['user_id']).count().reset_index()[['user_id','business_id']]

users_neigh_fict =     yl_users_neighbours.set_index('user_id').T.to_dict()

mean_neighbours_user = yl_users_neighbours['business_id'].mean()

with open(os.path.join(train_folder, 'business.json') ,'r') as photo:
    photo_data = photo.readlines()

postal_avg = {}

for i in photo_data:
    key = json.loads(i.replace("\n",''))['postal_code']
    val = float(json.loads(i.replace("\n",''))['stars'])
    pair = postal_avg.get(key,(0,0))
    postal_avg[key] = (pair[0]+val,pair[1]+1)


postal_avg_1 = {}
for key,val in postal_avg.items():
    postal_avg_1[key] = val[0]/val[1]

business_postal_avg = {}

for i in photo_data:
    key = json.loads(i.replace("\n",''))['business_id']
    val = json.loads(i.replace("\n",''))['postal_code']
    business_postal_avg[key] = postal_avg_1[val]


# json.loads(photo_data[0].replace("\n",''))['business_id']

business_photo_cnt = {}

for i in photo_data:
    key = json.loads(i.replace("\n",''))['business_id']
    val = float(json.loads(i.replace("\n",''))['stars'])
    business_photo_cnt[key] = val


# json.loads(photo_data[0].replace("\n",''))['business_id']

'''
loading results from item based collaborative filtering to be used as a feature


'''



train_df = pd.read_csv("Train_data_pd.csv")
train_item_preds = yelp_train.copy()
train_item_preds['val'] = train_item_preds['business_id'].apply(func)

train_df['pred_item']= train_item_preds[['val']]
train_item_preds['val2'] = train_item_preds['business_id'].apply(func2)

train_df['pred_item2']= train_item_preds[['val2']]

train_item_preds['val3'] = train_item_preds['user_id'].apply(func3)

train_df['pred_item3']= train_item_preds[['val3']]


train_X = train_df[['x1','x2','x3','x4','x5','x6','x7','x8','pred_item','pred_item2','pred_item3']]

train_X['pred_item4']= train_item_preds['business_id'].apply(func4)

train_y = train_df[['target']]

dev_df = pd.read_csv("val_dataset_pd.csv")
dev_item_preds = pd.read_csv(test_file)
dev_item_preds.columns = ["user_id","business_id","stars"]
dev_item_preds['val'] = dev_item_preds['business_id'].apply(func)
    
dev_df['pred_item']= dev_item_preds[['val']]
dev_item_preds['val2'] = dev_item_preds['business_id'].apply(func2)

dev_df['pred_item2']= dev_item_preds[['val2']]

dev_item_preds['val3'] = dev_item_preds['user_id'].apply(func3)

dev_df['pred_item3']= dev_item_preds[['val3']]


dev_x = dev_df[['x1','x2','x3','x4','x5','x6','x7','x8','pred_item','pred_item2','pred_item3']]

dev_x['pred_item4']= dev_item_preds['business_id'].apply(func4)


# In[ ]:





# In[12]:



prediors = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7','pred_item','pred_item2','pred_item3','pred_item4']

import xgboost as xgb

import time

start = time.time()
model = xgb.XGBRegressor(learning_rate=0.04,n_estimators=1000,max_depth=6 )
# model.fit(X_1, train_y)

model.fit(train_X[prediors], train_y)
# model.fit(X, y)

# 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8','pred_item','pred_item2',
# learning_rate=0.05,n_estimators=2000,max_depth=5   0.9805

# 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8','pred_item','pred_item2',
# learning_rate=0.04,n_estimators=1000,max_depth=6 0.9803



# prediors = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7','pred_item','pred_item2']
# learning_rate=0.04,n_estimators=1000,max_depth=6 0.9803


# prediors = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7','pred_item','pred_item2','pred_item3']
# learning_rate=0.04,n_estimators=1000,max_depth=6 0.9801211

# prediors = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7','pred_item','pred_item2','pred_item3','pred2by3']
# learning_rate=0.04,n_estimators=1000,max_depth=6 0.9809211

# prediors = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7','pred_item','pred_item2','pred_item3','pred_item4']
# learning_rate=0.04,n_estimators=1000,max_depth=6 0.979839
import gc

gc.collect()


# In[13]:


predictions = model.predict(dev_x[prediors])

# labels = dev_y['target'].values
# print("Test rmse")

# print(rmse(predictions, labels))

end = time.time()

print("Time taken",(end-start)/60)

pred_modified = []

for i in predictions:
    if i<1:
        pred_modified.append(1)
    elif i>5:
        pred_modified.append(5)
    else:
        pred_modified.append(i)
pred_modified_array = np.array(pred_modified).reshape(-1,1)        



# In[14]:


# def rmse(predictions, targets):
#     return np.sqrt(((predictions - targets) ** 2).mean())


# In[ ]:


users= list(dev_item_preds['user_id'].values)

business= list(dev_item_preds['business_id'].values)


# In[ ]:






prediction = list(pred_modified_array)

tups = list(zip(users,business,prediction))

with open(output_file, 'w+') as task2_2_out:
    task2_2_out.write('user_id,  business_id, prediction\n')
    for i in range(len(tups)):
        user_id = tups[i][0]
        business_id = tups[i][1]
        prediction_val = str(tups[i][2][0])
        write_query = user_id+","+business_id+","+prediction_val
        if i==len(tups)-1:
            task2_2_out.write(write_query)
        else:
            task2_2_out.write(write_query)
            task2_2_out.write("\n")
task2_2_out.close()

endtime = time.time()

print("Time taken by entire process ----> ",endtime-start_time)




