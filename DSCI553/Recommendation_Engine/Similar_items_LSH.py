#!/usr/bin/env python
# coding: utf-8



import sys
import time

from itertools import combinations
import math

from operator import add
from pyspark import SparkContext

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]



sc = SparkContext.getOrCreate()


sc.setLogLevel("WARN")

# In[3]:


Number_hash_fxns = 60



start = time.time()
### load data into rdd and do rough preprocessing
input_data = sc.textFile(input_file_path).filter(lambda x :x !='user_id,business_id,stars' ).                                            map(lambda x : x.split(",")).map(lambda x : (x[0],x[1],float(x[2])))


### Aim is identify similar businesses 
'''
Pseudo Code

Basically similar to lecture notes, business ~ Documents 
                                    User Ids ~ Shingles
                                
We convert all user_ids aka shingles into outputs of certain hash functions converting them into integers from 0 to 11270 or any other hashing method

Now we generate a Matrix with users on Row Index and Business IDs on COlumn index ;

Size of Matrix : 11270 X 24732 Wide

Too Big

### Gnerate Singature Matrix using some hash functions, I think Hash functions given in Assignment document

Lets assume we use M hash functions using the concept f(x)= (ax + b) % m or f(x) = ((ax + b) % p) % m

Now we will have a Signature Matrix of Size M X 24732

Now, we use LSH to divide matrix into bXn chunks and using hash functions see if two columns get into same bucket.

Getting atleast into same bucket makes a column pair as Candiate Pair

For these candidate pairs, calculate actual jaccard Similarity and write these similairtues >= 0.5 into output file 

'''



user_indexs_tmp = input_data.map(lambda x : x[0]).distinct().sortBy(lambda k : k)


# In[14]:


user_indexes = user_indexs_tmp.zipWithIndex()



# user_indexes.take(5)
# [('---1lKK3aKOuomHnwAkAow', 0),
#  ('--2vR0DIsmQ6WfcSzKWigw', 1),
#  ('--BumyUHiO_7YsHurb9Hkw', 2),
#  ('--Qh8yKWAvIP4V4K8ZPfHA', 3),
#  ('--RlSfc-QmcHFGHyX6aVjA', 4)]
# business_indexes




## Business Indexing 
business_indexs_tmp = input_data.map(lambda x : x[1]).distinct().sortBy(lambda k : k)
business_indexes = business_indexs_tmp.zipWithIndex()


# business_indexes.take(5)
# [('--6MefnULPED_I942VcFNA', 0),
#  ('--7zmmkVg-IMGaXbuVd0SQ', 1),
#  ('--9e1ONYQuAa-CB_Rrw7Tw', 2),
#  ('--DaPTJW3-tB1vP-PfdTEg', 3),
#  ('--FBCX-N37CMYDfs790Bnw', 4)]



## Now lets generate a charateristic Matrix 

'''
We first need to find list of all the users who voted for a given business.

Business ID : [ LIst of User ids { Numeric } who even gave a star rating]

users who didnt give vote , we dont care about them. lets say for 

[ 0 ,[1,3,67,123,5678]]

Now, we generate their indexing based on M hash functions 

Lets say, we call hash function H1 on [1,3,67,123,5678] and their indexes change to [56,34,12,566,78] then based on smallest index among these, we define first element of our signature matrix

we do it for all hash functions for a given business ID and we get 1 column of signature matrix - Basically key value pair with key as business id and value as list of indexes of signatures


'''

# https://stackoverflow.com/questions/31006438/extracting-a-dictionary-from-an-rdd-in-pyspark/31107894


user_indexes_dict = user_indexes.collectAsMap()


business_indexes_dict = business_indexes.collectAsMap()




### now,we need to generate list of users who reviewed a business

Business_Rated_users = input_data.map(lambda x : (x[1],x[0])).groupByKey().mapValues(set).map(lambda x : (    business_indexes_dict[x[0]],sorted([user_indexes_dict[w] for w in list(x[1])])))
# input_data.map(lambda x : (x[1],x[0])).groupByKey().mapValues(set).map(lambda x : (\
#     x[0],
#     [w for w in list(x[1])])
#     ).take(1)
# [('3MntE_HWbNNoyiLGxywjYA',
#   ['kjaUSiRWhR9bF9KxOMbVvg',
#    'TQXtrSpsUyvHMriX8hvNWQ',
#    '4o0KkpAkyO6r0NHXmobTeQ',
#    'xhlcoVm3FOKcxZ0phkdO6Q',
#    'T13IBpJITI32a1k41rc-tg'])]
# [(1715, [8600, 5367, 1051, 10883, 5300])]

# ## now for each business, We have list of all the user ids who ever voted for that busines. Think of each row in this rdd as 1 column 
# ## Each column only contains characteristic matrix row numbers which have value = 1




### Now is time to build Signature Matrix. For that, we need to build set of Hash Functions 

# f(x) = ((ax + b) % p) % m
# where p is any prime number and m is the number of bins
'''
Now we need to generate Number_hash_fxns functions

for generating a hash function, we need a,b randomly generated and p we choose a prime number, m is number of bins

Read lecture notes to see how a hash function is used in generating derived indexes

'''


# In[22]:


import numpy





##https://stackoverflow.com/questions/49631178/using-for-loop-to-define-multiple-functions-python

##https://www.freecodecamp.org/news/prime-numbers-list-chart-of-primes/

a_vector = numpy.random.randint(1,10000000,Number_hash_fxns)
b_vector = numpy.random.randint(1,10000000,Number_hash_fxns)

m = [len(user_indexes_dict)+len(business_indexes_dict)]*Number_hash_fxns

def addfunc(a,b,m):
    def hash_func_n(x):
        return ((a*x + b) % 10883) % m
    return hash_func_n

function_list = []
for i in zip(a_vector,b_vector,m):
    function_list.append(addfunc(i[0],i[1],i[2]))

# 
def find_signature_val(sorted_input,hash_output_list_1):
    signature_vals = []
    for haslist1 in  hash_output_list_1:
        sig_value = sorted(list(zip(sorted_input,haslist1)), key=lambda x : x[1])[0][0]
        signature_vals.append(sig_value)
    return signature_vals
    




def hasfunction_apply(x):
    sorted_input = x
    hash_output_list_1= []
    for i in range(Number_hash_fxns):
        hash_output_list = []
        for j in sorted_input:
            hash_output_list.append(function_list[i](j))
        hash_output_list_1.append(hash_output_list)
    return find_signature_val(sorted_input,hash_output_list_1)




Business_Rated_users_repart = Business_Rated_users.repartition(100)


# In[31]:


Signature_Matrix = Business_Rated_users_repart.mapValues(hasfunction_apply)


# In[32]:






import itertools




import math
def convert_sig2_b_n_LSH(sig_vector, num_bands):
    bands_list = []
#     TypeError: unhashable type: 'list'
    number_rows = math.ceil(len(sig_vector)/num_bands)
#     print(number_rows)
    cnt = 0
    for i in range(0,len(sig_vector),number_rows):
        target_vector = sig_vector[i:i+number_rows]
#         TypeError: unhashable type: 'list'
        hash_target = hash(tuple(target_vector))
#         print(i)
        bands_list.append((cnt,hash_target))
        cnt+=1
    return bands_list





from itertools import combinations

def conbinations_sort(lst):
    return [businessid1_id2 for businessid1_id2 in combinations(sorted(list(set(lst))), 2)]




candidate_pairs = Signature_Matrix         .flatMap(lambda x: [(tuple(n_bins), x[0]) for n_bins in convert_sig2_b_n_LSH(x[1], 30)])         .groupByKey().map(lambda x: list(x[1])).filter(lambda list_cnt: len(list_cnt) > 1)         .flatMap(conbinations_sort).filter(lambda x : x[0]!=x[1]).distinct()




first_join = candidate_pairs.leftOuterJoin(Business_Rated_users)



both_joined = first_join.map(lambda x : (x[1][0],(x[0],x[1][1]))).leftOuterJoin(Business_Rated_users)


# In[43]:


join_formatted = both_joined.map(lambda x :((x[0],x[-1][0][0]),(x[-1][-1],x[-1][0][1])))




def above_threshold(jaccard,thresh=0.5):
    if jaccard>=0.5:
        return 1
    else:
        return 0
    

def jaccard_similarity(vector1,vector2):
    '''
    essentiallly jacard similairty can be solved using Venn Diagrams
    Jaccard_simialrty = (Vector1 Intersect Vector2)/ ( Vector1 Union Vector2))
    '''
    set1 = set(vector1)
    set2 = set(vector2)
    
    jaccard = len((set1 & set2))/len((set1 | set2))
    
#     if above_threshold(jaccard):
        
    
    return jaccard




output_pairs_list = join_formatted.map(lambda x : (x[0],jaccard_similarity(x[1][0],x[1][1]))).filter(lambda x : (x[1]>=0.5))#.sortBy(lambda x : -x[1])



index_2_bns_dict = {}

for key,val in business_indexes_dict.items():
    index_2_bns_dict[val] = key



output_pairs_list_export = output_pairs_list.map(lambda x : ((index_2_bns_dict[x[0][0]],index_2_bns_dict[x[0][1]]),x[1]))

sorted_output_pairs = output_pairs_list_export.map(lambda x : ((sorted(x[0])),x[1])).map(lambda x : (x[0][0],x[0][1],x[1])).sortBy(lambda a : (a[0],a[1])).collect()

def format_output(x):
    return x[0]+","+x[1]+","+str(x[2])


with open(output_file_path, 'w+') as LSH_has_ouput:
    LSH_has_ouput.write("business_id_1, business_id_2, similarity\n")
    for index,w in enumerate(sorted_output_pairs):
        if index==len(sorted_output_pairs)-1:
            LSH_has_ouput.write(format_output(w))
        else:
            LSH_has_ouput.write(format_output(w)+"\n")
LSH_has_ouput.close()       


# In[55]:


#print(candidate_pairs.count())

print(time.time()-start)
