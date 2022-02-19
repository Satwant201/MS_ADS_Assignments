
'''
1) I check for items in frequent singleton set. 
1a) maintain a counter which tells which level of itmeset we r going to produce, 2/3/4/....n -- K 
2) output of 1 is my current iteration input df
3) generate k-1 level itemsets and filter out those not found in frequent_itemset dictionary with key value k-1
4) Output of 3) again flatmap and generate combinations of level k. filter out values below support.
5) store the output in frequent_itemset dictionary with key value k
6) repeat until the input DF becomes empty, then converge
'''



################ Task 1 Begins ##############

from pyspark import SparkContext

from operator import add
import sys

import json 


import time


from operator import add
from itertools import combinations


def f(x): return x

def combinations_feat(x,n):
    return [sorted(tuple(w)) for w in list(combinations(x,n))]

def A_priori_algorithm(key_item_rolled,support=1):
    item_counts = key_item_rolled.flatMapValues(f).map(lambda x : (x[1],1)).reduceByKey(add).filter(lambda x : x[1]>=support).map(lambda x : x[0]).collect()
    print("Number of Unique Singlet Frequent Itemsets {}".format(len(item_counts)))
    frequent_itemset= {}
    frequent_itemset[1] = item_counts
    if len(item_counts)==0:
        return {}
    k = 2
    while(1):
        if k==2:
            pass2_input_data= key_item_rolled.flatMapValues(f).filter(lambda x : x[1] in item_counts).groupByKey().mapValues(set).map(lambda x : (x[0],list(x[1])))
            candidate_pairs = pass2_input_data.map(lambda x : (x[0],combinations_feat(x[1],k))).flatMapValues(f).map(lambda x : (tuple(x[1]),1)).reduceByKey(add).filter(lambda x : x[1]>=support).map(lambda x : sorted(x[0])).collect()
            if len(candidate_pairs)==0:
                break
            else:
                frequent_itemset[k] = candidate_pairs
        else:
            pass2_input_data= key_item_rolled.flatMapValues(f).filter(lambda x : x[1] in item_counts).groupByKey().mapValues(set).map(lambda x : (x[0],list(x[1])))
            filtered_k_minus_1_itemset= pass2_input_data.map(lambda x: (x[0],combinations_feat(x[1],k-1))).flatMapValues(f).filter(lambda x : sorted(x[1]) in frequent_itemset[k-1])
            filtered_input_data_pass3 = filtered_k_minus_1_itemset.flatMapValues(f).groupByKey().mapValues(set).map(lambda x : (x[0],list(x[1])))
            candidate_pairs = filtered_input_data_pass3.map(lambda x : (x[0],combinations_feat(x[1],k))).flatMapValues(f).map(lambda x : (tuple(x[1]),1)).reduceByKey(add).filter(lambda x : x[1]>=support).map(lambda x : sorted(x[0])).collect()
            if len(candidate_pairs)==0:
                break
            else:
                frequent_itemset[k] = candidate_pairs
        print("Number of {} Wise Frequent Itemset {}".format(k,len(candidate_pairs)))
        k+=1
    return frequent_itemset

#Create SparkSession
sc = SparkContext.getOrCreate()

t1 = time.time()


input_json_path = "small2.csv"

case = 2


support = 9

output_file_path = "../Data/output1.txt"

input_data_tmp = sc.textFile(input_json_path)

input_data  = input_data_tmp.filter(lambda x : x!='user_id,business_id').\
                            map(lambda x : x.split(",")).\
                            map(lambda x : (x[0],x[1]))
if case==1:
    key_data = 0
    value_data = 1
else:
    key_data= 1
    value_data = 0


print("Case Number {}".format(case))


key_item_rolled = input_data.map(lambda x : (x[key_data],x[value_data])).groupByKey().mapValues(set).map(lambda x : (x[0],sorted(list(x[1]))))



Candidate_pairs_A_prio = A_priori_algorithm(key_item_rolled)


