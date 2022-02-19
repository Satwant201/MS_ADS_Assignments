'''
Steps I feel can help complete this assignment


######################### MAP 1 REDUCE 1 #######################################################
1) Load Data

2) Enumerate all possible itemsets in the underlying chunk of data

3) Data from 2) becomes input for In-Memory Frequent Itemset Algorithm. 

4) Design a FUnction for PCY/A Prior which takes in bucket- itemset and returns potential frequent itemsets in that data at given support level

5) Export potential candidates into reduce operation and just maybe count their frequency

################################################################################################

######################### MAP 2 REDUCE 2 ########################################################

1) Load subsets of data, for all outputs of Reduce 1, keep logging frequency of those candidate pairs

2) Take sum of occurances at itemset level as key and filter keys below threshold S

3) Export Output of Reduce operation, these r the frequent itemsets in the Data

'''


from itertools import combinations
import math
from operator import add
from pyspark import SparkContext
import sys
import time

case_number = int(sys.argv[1])
support = int(sys.argv[2])
input_file_path = sys.argv[3]
output_file_path = sys.argv[4]

sc = SparkContext.getOrCreate()

start = time.time()


input_data_tmp = sc.textFile(input_file_path)


input_data  = input_data_tmp.filter(lambda x : x!='user_id,business_id').\
                            map(lambda x : x.split(",")).\
                            map(lambda x : (x[0],x[1]))

if case_number==1:
    key_data = 0
    value_data = 1
else:
    key_data= 1
    value_data = 0

print("Case Number {}".format(case_number))


new_rdd = input_data.map(lambda x : (x[key_data],x[value_data])).groupByKey().mapValues(set).map(lambda x : (sorted(list(x[1]))))

key_item_rolled = input_data.map(lambda x : (x[key_data],x[value_data])).groupByKey().mapValues(set).map(lambda x : (x[0],sorted(list(x[1]))))



whole_size = new_rdd.count()



def Higher_N_Candidate_search(dataset, Lk, k,limit=100):
    Ck = {}
    for li in dataset:
        li = sorted(set(li) & set(Lk))
        #print('li:', li)
        for item in combinations(li, k):
            item = tuple(item)
            cnt = Ck.get(item,0)
            Ck[item]=cnt+1
    return Ck

def sort_list(l):
    return sorted(l)


def local_support(p,whole_size,support, global_support = 10):
    ratio = p / whole_size
    return math.ceil(ratio * support)


'''
1) I check for items in frequent singleton set. 
1a) maintain a counter which tells which level of itmeset we r going to produce, 2/3/4/....n -- K 
2) output of 1 is my current iteration input df
3) generate k-1 level itemsets and filter out those not found in frequent_itemset dictionary with key value k-1
4) Output of 3) again flatmap and generate combinations of level k. filter out values below support.
5) store the output in frequent_itemset dictionary with key value k
6) repeat until the input DF becomes empty, then converge
'''


def f(x): return x

def combinations_feat(x,n):
    return [sorted(tuple(w)) for w in list(combinations(x,n))]

def A_priori_algorithm(key_item_rolled):
    print("Support Value used {}".format(support))
    item_counts = key_item_rolled.flatMapValues(f).map(lambda x : (x[1],1)).reduceByKey(add).filter(lambda x : x[1]>=support).map(lambda x : x[0]).collect()
    print("Number of Unique Singlet Frequent Itemsets {}".format(len(item_counts)))
    frequent_itemset= {}
    frequent_itemset[1] = item_counts
    if len(item_counts)==0:
        return {}
    elif len(item_counts)>0:
        return frequent_itemset
    else:
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


def A_prior_data_prep(dataset):
    a = len(dataset)
    b = support
    try:
        ratio = a/b
    except:
        ratio=0
    return ratio



def A_priori_Algo(dataset, support, whole_size, max_n = 10):
    frequent_itemsets = []
    partition = list(dataset)
    p = len(partition)
    ps = local_support(p,whole_size,support)
    C1 = {}
    for li in partition:
        for element in li:
            cnt = C1.get(element,0)
            C1[element]=cnt+1
    Lk = []
    for key, val in C1.items():
        if val >= ps:
            Lk.append(key)
    L1 = sort_list(Lk)
    # return return_data
    #L1 = Above_support_pair_check(C1, ps)

    frequent_itemsets.append([(item,) for item in L1])
    k = 2
    while (1):
        _ = A_prior_data_prep(partition)

        Ck = Higher_N_Candidate_search(partition, L1, k)
        Lk = []
        for key, val in Ck.items():
            if val >= ps:
                Lk.append(key)
        Lk = sort_list(Lk)



        #Lk = Above_support_pair_check(Ck, ps)

        _ = len(Lk)

        if _==0:
            break
        if Lk == []:
            break
        frequent_itemsets.append(Lk)
        L1 = set()
        for item in Lk:
            L1 = L1 | set(item)
        ## increment the counter to move to next pass . 
        k =k+ 1
    return frequent_itemsets


def format_dictionary_2_list(result_dict):
    lst = []
    for k11,v11 in result_dict.items():
        lst.append((k11,v11))
    return lst


def SON_Map2_Task(dataset, candidates):
    result_dict = {}
    for li in dataset:
        for item in candidates:
            delta = set(item)
            if delta.issubset(li):
                cnt = result_dict.get(item,0)
                result_dict[item]=cnt+1
    result_li = format_dictionary_2_list(result_dict)
    return result_li

def SON_Reduce2_Task(df,support_level):
    frequent_itemsets_tmp_1 = df.reduceByKey(add)
    frequent_itemsets_tmp_2 = frequent_itemsets_tmp_1.filter(lambda x: x[1] >= support_level).map(lambda x: x[0])
    final_frequents = frequent_itemsets_tmp_2.sortBy(lambda x: (len(x), x)).collect()
    return final_frequents



def post_processing(data,result='',length=1):
    for item in data:
        ## check for singletons, there we need to fix (a), to (a)
        if len(item) == 1:
            result = result + str(item).replace(',', '') + ','
        elif len(item) == length:
            result =result+ str(item) + ','
        else:
            result =result+ '\n\n'
            result = result+str(item) + ','
            length = len(item)
    result = result.replace(',\n\n', '\n\n')[:-1]
    return result

def SON_Phase_1(new_rdd,support):
    candidates_tmp = new_rdd.mapPartitions(lambda partition: A_priori_Algo(partition, support, whole_size))
    candidates_tmp_1 = candidates_tmp.flatMap(lambda x: x).distinct()
    return candidates_tmp_1.sortBy(lambda x: (len(x), x)).collect()

def SON_Phase_2(new_rdd,support):
    global_ss = support
    frequent_itemsets_tmp = new_rdd.mapPartitions(lambda partition: SON_Map2_Task(partition, candidates))
    return frequent_itemsets_tmp





# Phase 1




candidates =  SON_Phase_1(new_rdd,support)

#print(candidates)
_ = A_priori_algorithm(key_item_rolled)

# Phase 2
frequent_itemsets_map_out  = SON_Phase_2(new_rdd,support)

frequent_itemsets = SON_Reduce2_Task(frequent_itemsets_map_out,support)

#print(frequent_itemsets)

with open(output_file_path, 'w+') as fout:
    fout.write('Candidates:\n' + post_processing(candidates) + '\n\n' + 'Frequent Itemsets:\n' + post_processing(frequent_itemsets))

end = time.time()

print('Duration: {}'.format(end - start))

