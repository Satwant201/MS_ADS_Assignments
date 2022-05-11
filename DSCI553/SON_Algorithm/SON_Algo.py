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

start = time.time()
filter_thresh = int(sys.argv[1])
support = int(sys.argv[2])
input_file_path = sys.argv[3]
output_file_path = sys.argv[4]

sc = SparkContext.getOrCreate()

task2_input = sc.textFile(input_file_path)

input_data  = task2_input.filter(lambda x : x!='"TRANSACTION_DT","CUSTOMER_ID","AGE_GROUP","PIN_CODE","PRODUCT_SUBCLASS","PRODUCT_ID","AMOUNT","ASSET","SALES_PRICE"').\
                            map(lambda x : x.split(",")).\
                            map(lambda x : (x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]))

# >>> input_data.take(1)
# [('"11/1/2000"', '"01104905"', '"45-49"', '"115"', '"110411"', '"4710199010372"', '"2"', '"24"', '"30"')]

processed_data = input_data.map(lambda x : (str(x[0][1:-1])+"-"+str(int(x[1][1:-1])),str(int(x[5][1:-1]))))

processed_data_tmp = input_data.map(lambda x : (str(x[0][1:-1])+"-"+str(int(x[1][1:-1])),str(int(x[5][1:-1])))).collect()


with open("customer_product.csv", 'w+') as finter:
    finter.write("DATE-CUSTOMER_ID, PRODUCT_ID\n")
    for index,item in enumerate(processed_data_tmp):
        if index!=len(processed_data_tmp)-1:
            finter.write(item[0]+","+str(item[1])+"\n")
        else:
            finter.write(item[0]+","+str(item[1]))



data_processing_aprior = processed_data.groupByKey().filter(lambda x : len(x[1])>filter_thresh).mapValues(set).map(lambda x : (sorted(list(x[1])))).repartition(4)


key_item_rolled = processed_data.groupByKey().filter(lambda x : len(x[1])>filter_thresh).mapValues(set).map(lambda x : (x[0],sorted(list(x[1])))).repartition(4)





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





def A_priori_Algo(Partition_data_process_sample, Global_Support_value, overall_data_process_size, max_n = 10):
    
    partition = list(Partition_data_process_sample)  ### iterator needs to be conveted to alist to be used
    p = len(partition)  ## we need to calucu;ate size of current partition 
    ps = local_support(p,overall_data_process_size,Global_Support_value)
    Pass_1_Dict = {}  # this dictionary will contain frequency of items k = 1
    for li in partition: ## we iterate through each bucket
        for element in li: ## for each item, we increase its count in the dictionary 
            cnt = Pass_1_Dict.get(element,0) ### if key is not found in dictionayr, initialize with 0 
            Pass_1_Dict[element]=cnt+1 ## increment the counter 
    #_ = get_key_bucket_freq(partition)
    Lk = set()   ### this is the list which will contai our frequent itemsets 
    '''
    basically in the above step, we calculated for pass 1, frequency of all bucket_members in the data_process
    now, we will apply local threshold on the the frequencies and identify k=1 candidate pairs
    now these candidate pairs will be used in subsequent steps in filtering out impossible pairs using the 
    principle of monotonicity and other laws governing frequent bucket_memberset theory
    '''
    for key, val in Pass_1_Dict.items():  ### basically we iterate though all the dictioney items 
        if val >= ps:  ### if frequency is above local support 
            Lk.add(key)  ### add to the frequent item list LK
    L1 = sort_list(list(Lk))  ### sorting is important, sort the list and save it in L1. L1 would be used later 
    
    potential_bucket_members_freq = []   ### we need to do some postprocessig of the tuples for later use 
    list_bucket_members_c1 = []  ### 
    for bucket_member in L1:
        list_bucket_members_c1.append((bucket_member,))

    #list_bucket_members_c1 = [(bucket_member,) for bucket_member in L1]
    
    potential_bucket_members_freq.append(list_bucket_members_c1)
    '''
    1) Pseudo code for future passed in the A priori Algorithm
    2) output of 1 is my current iteration input df
    3) generate k-1 level bucket_membersets and filter out those not found in frequent_bucket_memberset dictionary with key value k-1
    4) Output of 3) again flatmap and generate combinations of level k. filter out values below Global_Support_value.
    5) store the output in frequent_bucket_memberset dictionary with key value k
    6) repeat until the input DF becomes empty, then converge
    '''
    k = 2
    while (1):
        _ = A_prior_data_process_prep(partition)


        Ck = Higher_N_Candidate_search(partition, L1, k)
        
        Lk = set()
        
        L1 = set()
        #_ = get_key_bucket_freq(partition)
# '''
# basically in the above step, we calculated for pass k, frequency of all bucket_members in the data_process
# now, we will apply local threshold on the the frequencies and identify k=k candidate pairs
# now these candidate pairs will be used in subsequent steps in filtering out impossible pairs using the 
# principle of monotonicity and other laws governing frequent bucket_memberset theory
# '''
        for key, val in Ck.items():
            if val >= ps:
                Lk.add(key)
        Lk = sort_list(list(Lk))
        _ = len(Lk)
        if _==0:
            break
        if Lk == []:
            break
        potential_bucket_members_freq.append(Lk)
        
        for bucket_member in Lk:
            L1 = L1 | set(bucket_member)
        k =k + 1
    return potential_bucket_members_freq



# def format_dictionary_2_list(result_dict):
#     lst = []   ### initialze is list 
#     for k11,v11 in result_dict.items(): ## iterate though output dictionary items 
#         lst.append((k11,v11))  ### make tuple of key and value and store in list 
#     return lst
def format_dictionary_2_list(result_dict):
    lst = set()   ### initialze is list 
    for k11,v11 in result_dict.items(): ## iterate though output dictionary items 
        lst.add((k11,v11))  ### make tuple of key and value and store in list 
    return list(lst)



def A_prior_data_process_prep(Partition_data_process_sample):
    '''
    bascially we are trying to estimate if we have enough data_process for generating value frequnet bucket_membersets 
    '''
    a = len(Partition_data_process_sample)
    b = support
    try:
        ratio = a/b
    except:
        ratio=0
    return ratio



# def format_dictionary_2_list(result_dict):
#     lst = []
#     for k11,v11 in result_dict.items():
#         lst.append((k11,v11))
#     return lst


def SON_Map2_Task(dataset, candidates):
    '''
    @Input : Dataset
    Candiates : Frequent itemset found in phase 1 of SON using subsets of data

    @Return Type:
    Reutn list of tuples with key, value depicting frequncy of each candiate in given partition 
    '''
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
    '''
    @Input : Dataframe, Support level
    Here basically start adding values for each key
    using global support filter, filter out infrequent keys
    Collect frequnt keys and store them in a list 
    
    @return : List of frequent itemsets in overall data 
    '''
    frequent_itemsets_tmp_1 = df.reduceByKey(add)
    frequent_itemsets_tmp_2 = frequent_itemsets_tmp_1.filter(lambda x: x[1] >= support_level).map(lambda x: x[0])
    final_frequents = frequent_itemsets_tmp_2.sortBy(lambda x: (len(x), x)).collect()
    counter_dict = {}
    for item in final_frequents:
        cnt = counter_dict.get(len(item),0)
        counter_dict[len(item)]=cnt+1
    return final_frequents

def post_processing(data_process,type_1,result='',length=1,separation_var ='\n\n' ):
    if type_1=='Candidate':
        print("Printing Potential Candidate Pair Counts for each K")
    else:
        print("Printing Final Frequent Itsemset Pair Counts for each K")
    cnt = {}
    for i in data_process:
        l = len(i)
        n1 = cnt.get(l,0)
        cnt[l]=n1+1
    tuple_list = set()
    for key,vali in cnt.items():
        tuple_list.add((key,vali))
    for bucket_member in data_process: ### for each frequent pair, iterate
        if len(bucket_member) == 1: ### check if its a singlet 
            a = 'singlets'
            legn = len(bucket_member)
            processed_name = str(bucket_member).replace(',', '')  ### in singlets, we had output like (val,) but we need to fix it to (val) format
            result = result + processed_name + ','  ### while iterating, data needs to comma separated
        elif len(bucket_member) == length: ### this for handling pass k itemsets 
            b = 'doublet'
            legn = len(bucket_member) ## we define new length varibale for future use
            processed_name = str(bucket_member)
            result =result+ processed_name + ','
        else:
            result =result+ separation_var   #### when length paramter is not met, then we need to add two new line characters to start next pass
            processed_name = str(bucket_member)
            result = result+processed_name + ','
            length = len(bucket_member)   ### now length variable gets updated to k. next iertation would use elif block 
    result = result.replace(',\n\n', separation_var)[:-1] ## there is always an extra comma at end of each pass, needs to be fixed and we also need to ignore last comma hence -1 index for slicing data
    return result

def SON_Phase_1(new_rdd,support):
    candidates_tmp = new_rdd.mapPartitions(lambda partition: A_priori_Algo(partition, support, whole_size))
    candidates_tmp_1 = candidates_tmp.flatMap(lambda x: x).distinct()
    return candidates_tmp_1.sortBy(lambda x: (len(x), x)).collect()

def SON_Phase_2(new_rdd,support):
    global_ss = support
    frequent_itemsets_tmp = new_rdd.mapPartitions(lambda partition: SON_Map2_Task(partition, candidates))
    return frequent_itemsets_tmp





whole_size = data_processing_aprior.count()



candidates =  SON_Phase_1(data_processing_aprior,support)

potential_candiates_print = post_processing(candidates,type_1 = "Candidate")

    
# Phase 2
frequent_itemsets_map_out  = SON_Phase_2(data_processing_aprior,support)



frequent_itemsets = SON_Reduce2_Task(frequent_itemsets_map_out,support)

final_frequent_itemset_str = post_processing(frequent_itemsets,type_1 = "Frequents")
#print(frequent_itemsets)

with open(output_file_path, 'w+') as fout:
    fout.write('Candidates:\n' + potential_candiates_print + '\n\n' + 'Frequent Itemsets:\n' + final_frequent_itemset_str )

end = time.time()

runtime = end-start

print('Duration: {}'.format(runtime))
