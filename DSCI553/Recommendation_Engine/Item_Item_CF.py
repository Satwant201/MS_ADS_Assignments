#!/usr/bin/env python
# coding: utf-8

# ## Case 1: Item-based CF recommendation system with Pearson similarity (2 points)

# In[1]:


import sys
import time

from itertools import combinations
import math

from operator import add
from pyspark import SparkContext,SparkConf

#input_file_path = "../Data/yelp_train.csv"

input_file_path=sys.argv[1]


# output_file_path = sys.argv[4]

#sc = SparkContext.getOrCreate()

# conf = SparkConf().setMaster("local") \
#         .setAppName("Satwant Singh Task2.1") \
#         .set("spark.executor.memory", "4g") \
#         .set("spark.driver.memory", "4g")\
#          .set("spark.driver.host", "localhost")
sc = SparkContext()#conf=conf)

sc.setLogLevel("WARN")
# In[2]:


start = time.time()


# In[3]:


input_data_tmp = sc.textFile(input_file_path)


# In[4]:


N = 50

import gc


# In[5]:


input_data_tmp.take(1)


# In[6]:


#header = input_data_tmp.take(1)

input_data = input_data_tmp.filter(lambda x : x!='user_id,business_id,stars').map(lambda x : x.split(",")).map(lambda x: (x[0],x[1],float(x[2]))).repartition(N)


# In[7]:


business_indexs_tmp = input_data.map(lambda x : x[1]).distinct()
business_indexes = business_indexs_tmp.zipWithIndex()


user_indexs_tmp = input_data.map(lambda x : x[0]).distinct()
user_indexes = user_indexs_tmp.zipWithIndex()


user_indexes_dict = user_indexes.collectAsMap()


business_indexes_dict = business_indexes.collectAsMap()


# In[8]:


business_index_toname_dict = {}
for k,v in business_indexes_dict.items():
    business_index_toname_dict[v] = k


# In[9]:


'''
Steps I feel will help me build Item-Item Based CF Engine

1) Make a Utility Matrix for Business and their ratings by different users. here Business is analogous to an Item in slides

2) Now for the items, using |N| Neighbours, 

'''


# In[10]:


# two business get some predictions from different users. 

# 1) generate average for each item
# 2) then generate user long vectors, wherever 0 impute with 0 for the sake of ease of calculation
# 3) Generate pearson Correlation for all item pairs. 
# 4) output of 3 becomes a lookup table for prediction task
# 5) For a given user, use weighted average formula for generating ratings


# In[11]:


# '''
# As Suggested by Prof Shen, I would first build my forumla and test them on Slide questions and once I get them correct, then only I will use 
# the main yelp_train file to generate recommendations 

# 1) Convert Utility Matrix in Slide 45 to a RDD Dataframe.

# 2) COmpute column averages and store them

# 3) Now built Pearson Correlation formula and compare answers with slidea and quiz questions coz I got them all right :)

# 4) Replace input file with yelp_train :)
# '''

# sample_dataset = [['U1','I1',2],['U2','I1',3],['U4','I1',5],
#                   ['U1','I2',1],['U3','I2',4],['U4','I2',3],
#                   ['U2','I3',5],['U3','I3',2],['U4','I3',1],
#                   ['U1','I4',3],['U2','I4',2],['U3','I4',3]
#                  ]


# In[12]:


gc.collect()


# In[13]:


sample_rdd = input_data.map(lambda x : (user_indexes_dict[x[0]],business_indexes_dict[x[1]],x[2]))


# In[14]:
sample_rdd.persist()


# In[17]:


sample_rdd.take(10)



user_business_rating_lookup = sample_rdd.map(lambda x : (x[0],(x[1],x[2]))).groupByKey().mapValues(list).collectAsMap()


# In[15]:


user_business_rating_lookup_1 = {}

for key,val in user_business_rating_lookup.items():
    tmp_dct = {}
    for vals in val:
        tmp_dct[vals[0]]= vals[1]
    user_business_rating_lookup_1[key] = tmp_dct
    
    


# In[16]:




# In[18]:


#Item_level_averages = sample_rdd.map(lambda x : (x[1],x[2])).groupByKey().mapValues(list).map(lambda x : (x[0],sum(x[1])/len(x[1]))).collectAsMap()


# In[19]:



Item_level_averages_rdd = sample_rdd.map(lambda x : (x[1],x[2])).groupByKey().mapValues(list).map(lambda x : (x[0],sum(x[1])/len(x[1])))


# In[20]:


# {'I2': 2.6666666666666665,
#  'I1': 3.3333333333333335,
#  'I4': 2.6666666666666665,
#  'I3': 2.6666666666666665}


# In[21]:




#Item_level_vectors = sample_rdd.map(lambda x : (x[1],(x[0],x[2]))).groupByKey().mapValues(list)


# [('I2', [('U1', 1), ('U3', 4), ('U4', 3)]),
#  ('I1', [('U1', 2), ('U2', 3), ('U4', 5)])]


# In[22]:


#sample_rdd.take(1)


# In[23]:


# left = sample_rdd.map(lambda x : (x[0],(x[1],x[2])))

left = sample_rdd.map(lambda x : (x[1],(x[0],x[2]))).join(Item_level_averages_rdd).                map(lambda x : (x[1][0][0],(x[0],round(x[1][0][1]-x[1][1],3))))


# In[24]:


# reformat_data = left.join(left).filter(lambda x : x[1][0][0]!=x[1][1][0])

reformat_data = left.join(left).filter(lambda x : x[1][0][0]>x[1][1][0])


# In[ ]:





# In[25]:



# reformat_data.map(lambda x: (tuple(sorted((x[1][0][0],x[1][1][0]))),(x[1][0][1],x[1][1][1]))).groupByKey().mapValues(list).take(100)

# item_item_rating_vectors = reformat_data.map(format_tuple).distinct().map(lambda x : (x[0],x[1])).groupByKey().mapValues(list)

item_item_rating_vectors = reformat_data.map(lambda x : ((x[1][0][0],x[1][1][0]),(x[1][0][1],x[1][1][1]))).groupByKey().mapValues(list)



# .groupByKey().flatMapValues(list).take(100)


# In[26]:


#item_item_rating_vectors.take(1)


# In[27]:


from math import sqrt as msqrt


# In[28]:


def pearson_Correlation_Calculation(x):
    Numerator = 0
    denomitor1 = 0
    denomitor2 = 0
    if len(x[1])==0:
        return (x[0],0)
    for pairs in x[1]:
        Numerator+=pairs[0]*pairs[1]
        denomitor1+=pairs[0]*pairs[0]
        denomitor2+=pairs[1]*pairs[1]
    
    if denomitor1==0:
        return (x[0],0)
    elif denomitor2==0:
        return (x[0],0)
    elif len(x[1])<50: ##  15(1.11) changed to 20(1.10) change to 25( 1.09) changed to 40(1.08) changed to 50 (1.07) change to 60.
        return (x[0],0)
    else:
        return (x[0],round(Numerator/(msqrt(denomitor1)*msqrt(denomitor2)),3))



Pearson_lookup_table = item_item_rating_vectors.filter(lambda x : len(x[1])>50).map(pearson_Correlation_Calculation).filter(lambda x : x[1]!=0).collectAsMap()

#Pearson_lookup_table = item_item_rating_vectors.map(pearson_Correlation_Calculation).collectAsMap()


training_over= time.time()


print("Training Time Taken : ",training_over-start)


#exit()



gc.collect()


# ## generate Weighted Average rating for a given user-business pair

# In[ ]:

test_file_path = sys.argv[2]


#prediction_data = sc.textFile("../Data/yelp_val.csv")

repart_num = 1   # at default partition, runtime is 60-80 seconds

prediction_data = sc.textFile(test_file_path).repartition(repart_num)





header = prediction_data.take(1)




#user_business_rating_lookup_1

def weighted_avg_prediction(prediciton_sample):
    try:
        user = user_indexes_dict.get(prediciton_sample[0],None)
        if user is None:
            return 3

        item= business_indexes_dict.get(prediciton_sample[1],None)

    #     itemset = perason_lookup_table_dict[item]
        numerator = 0
        denominator = 0
        rat_tuples = []
        weight_lst = set()
        rating_lookup_dict = user_business_rating_lookup_1[user]
        if item is None:
            rating = 0
            for value in rating_lookup_dict.values():
                rating+=value
            return rating/len(rating_lookup_dict)

            
#             return NaN

        for item_2,rating in rating_lookup_dict.items():
            key = tuple(sorted((item,item_2),reverse=1))
            weight = Pearson_lookup_table.get(key,0.00000001) #change 0.00000001 to 0
    #         print(weight)
            numerator+=weight*rating
            denominator+=abs(weight)
            if weight!=0.00000001:
                rat_tuples.append((weight,rating))
                weight_lst.add(weight)
        if len(rat_tuples)==0 or list(weight_lst)==[0] :
            rating = 0
            for value in rating_lookup_dict.values():
                rating+=value
            return rating/len(rating_lookup_dict)
#         print(numerator/denominator)
        sorted_tuples = sorted(rat_tuples,key=lambda x : -x[0])[:5]
        
#         print(len(sorted_tuples))
        numerator1 = 0
        denominator1 = 0

        for items in sorted_tuples:
            numerator1+=items[0]*items[1]
            denominator1+=abs(items[0])
            
        if denominator1==0:
            rating = 0
#             print("whoo")
            for value in rating_lookup_dict.values():
                rating+=value
            return rating/len(rating_lookup_dict)
        if numerator1<0:  ### added this line to remove negative ratings 
            return  3     #### as of now best score found at 3. change to abs() error increased
        else:
            if (numerator1/denominator1)>5: ### added this line to put uppe rlimit to 5
                return 5
            else:
                return (numerator1/denominator1)
    except:
        return 3


pred_dat_input = prediction_data.filter(lambda x : x!=header[0]).map(lambda x : x.split(","))


# In[59]:


#pred_dat_input.take(1)


# In[ ]:


s1 = time.time()
pred_output = pred_dat_input.map(lambda x : (x[0],x[1],round(weighted_avg_prediction((x[0],x[1])),2))).collect()
s2 = time.time()
print(s2-s1)

print("Scoring Time Taken : ",(s2-training_over))



# In[ ]:



def format_output(x):
    return x[0]+","+x[1]+","+str(x[2])



output_file_path= sys.argv[3]
with open(output_file_path, 'w+') as LSH_has_ouput:
    LSH_has_ouput.write("user_id, business_id, prediction\n")
    for index,w in enumerate(pred_output):
        if index==len(pred_output)-1:
            LSH_has_ouput.write(format_output(w))
        else:
            LSH_has_ouput.write(format_output(w)+"\n")
LSH_has_ouput.close()       


exit()

