#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

import json 


import time



start = time.time()


# cluster_input_file = "../Data/hw6_clustering.txt"
# num_cluster = 10
# output_file_path = "../Data/tas1output12.txt"

cluster_input_file = sys.argv[1]
num_cluster = int(sys.argv[2])
output_file_path = sys.argv[3]


# Task: Passed Successfully                                                                                                             
#     Accuracy:  0.9997828191317729                                                                                                         
#     Time Taken: 71.84208035469055   


# In[2]:


with open(cluster_input_file,"r") as fin:
    input_data_tmp = fin.readlines()
    input_data_tmp2 = [w.replace("\n",'').split(",") for w in input_data_tmp]
    input_data = [[float(w1) for w1 in w] for w in input_data_tmp2]
fin.close()


# In[3]:


print("Number of Records in Input Data {}".format(len(input_data)))
# input_data
print("Number of Features in Each Data point {}".format(len(input_data[0][2:])))


# In[4]:


from sklearn.cluster import KMeans
import numpy as np


# In[5]:


# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# 
# >>> 
# >>> X = np.array([[1, 2], [1, 4], [1, 0],
# ...               [10, 2], [10, 4], [10, 0]])
# >>> kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
# >>> kmeans.labels_
# array([1, 1, 1, 0, 0, 0], dtype=int32)
# >>> kmeans.predict([[0, 0], [12, 3]])
# array([1, 0], dtype=int32)
# >>> kmeans.cluster_centers_
# array([[10.,  2.],
#        [ 1.,  2.]])


# In[6]:


global val2Index_dict,val2Clusterid_actual
val2Index_dict= {}
val2Clusterid_actual = {}
for i in range(len(input_data)):
    val = str(input_data[i][2:])
    val2Index_dict[val]=input_data[i][0]
    val2Clusterid_actual[val] = input_data[i][1]
    
    
    


# In[25]:


# initialization_results 
# 
global intemediate_export_dict
intemediate_export_dict = {}
intemediate_export_dict[0] = "The intermediate results:\n"

def intermediate_result_report(epoch,rs_cnt):

    DS_cnt = 0
    for key, val in cluster_DS_Stats_dict.items():
    #     print(cluster_DS_Stats_dict[key]['N'])
        DS_cnt+=len(cluster_DS_Stats_dict[key]['Members_index'])
    #     print(key)

    CS_cnt = 0
    CS_key_cnt = 0
    for key, val in cluster_CS_Stats_dict.items():
    #     print(val['N'])
        CS_cnt+=len(cluster_CS_Stats_dict[key]['Members_index'])
        CS_key_cnt+=1
    #     print(key)

    DS_point_cnt =DS_cnt
    CS_points_cnt = CS_cnt
    CS_cluster_cnt = CS_key_cnt
    RS_cnt = rs_cnt

    print("Number of Points in DS : {}".format(DS_point_cnt))
    print("Number of Clusters in CS : {}".format(CS_cluster_cnt))
    print("Number of Points in CS : {}".format(CS_points_cnt))
    print("Number of Points in RS : {}".format(RS_cnt))

    intemediate_export_dict[epoch+1] = "Round {}: {},{},{},{}\n".format(epoch+1,DS_point_cnt,CS_cluster_cnt,CS_points_cnt,RS_cnt)


# In[26]:


'''
For each new datapoint in new load, 

calculate mahanalobis distance between new point and if the distance is less than 2*sqrt(d)

then add the datapoint to the cluster with lowest distance

else do the same for compression set.

else put data point in RS
'''


# In[27]:


global cluster_DS_Stats_dict,cluster_CS_Stats_dict
global RS_set_iter1

# def Initizlization():
X = [w[2:] for w in input_data]
y = [w[1] for w in input_data]

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
# X, y, test_size=0.20, random_state=42)

# https://stackabuse.com/how-to-randomly-select-elements-from-a-list-in-python/
# https://stackabuse.com/how-to-randomly-select-elements-from-a-list-in-python/

### Step 1 
import random 

X_array = np.array(input_data)

_20prct = int(0.2*len(input_data))

epoch = 0

random20prct_x = X[0:_20prct]

print("Number of records in sample 20% data : {} ".format(len(random20prct_x)))

### Step 2Run K-Means (e.g., from sklearn) with a large K (e.g., 5 times of the number of the input clusters)

step2_k = 5*num_cluster

kmeans_step2 = KMeans(n_clusters=6*num_cluster, random_state=1201,).fit_predict(random20prct_x)

#Task: Passed Successfully
#Accuracy: 0.9997021519521457
#Time Taken: 79.56037425994873

#Task: Passed Successfully
#Accuracy: 0.9999751793293454
#Time Taken: 77.87102317810059

cluster_label = list(kmeans_step2)

_cluster_dict = {}
for i in cluster_label:
    cnt = _cluster_dict.get(i,0)
    _cluster_dict[i] = cnt+1

RS_clster = dict()
for k,v in _cluster_dict.items():
    if v==1:
        RS_clster[k] = v

RS_clster_list = list(RS_clster.keys())


# Step 3. In the K-Means result from Step 2, move all the clusters that contain only one point to RS
# (outliers).
DS_data = []
RS_set = []
for i in range(len(random20prct_x)):
    if cluster_label[i] not in RS_clster_list:
        DS_data.append(random20prct_x[i])
    else:
        RS_set.append(random20prct_x[i])




# Step 4. Run K-Means again to cluster the rest of the data points with K = the number of input clusters.

kmeans_step4 = KMeans(n_clusters=num_cluster, random_state=1201,).fit_predict(np.array(DS_data))




###Step 5. Use the K-Means result from Step 4 to generate the DS clusters (i.e., discard their points and
# generate statistics).
cluster_dict_DS = {}
for i in range(len(kmeans_step4)):
    clutser_id = kmeans_step4[i]
    clutser_list = cluster_dict_DS.get(clutser_id,[])
    clutser_list.append(DS_data[i])
    cluster_dict_DS[clutser_id] = clutser_list


###lets generate statistics for these DS clusters 
cluster_DS_Stats_dict = {}
for key,val in cluster_dict_DS.items():
    data_points = np.array(val)
    val_indexes = []
    for row in val:
        row_str = str(row)
        val_indexes.append(int(val2Index_dict[row_str]))

    sum_n = np.sum(data_points,axis=0)
    sumsq = np.sum(np.multiply(data_points,data_points),axis=0)
    centroid = sum_n/len(val)
    N = len(val)
    std = np.sqrt(sumsq/N - np.square(sum_n/N))
    cluster_DS_Stats_dict[key] = {"N":N,"sum_n":sum_n,"sumsq":sumsq,"centroid":centroid,"std":std,"Members_index":val_indexes}






# Step 6. Run K-Means on the points in the RS with a large K (e.g., 5 times of the number of the input
# clusters) to generate CS (clusters with more than one points) and RS (clusters with only one point).

print("Number of Points in RS after first Iter : {}".format(len(RS_set)))

kmeans_step6 = KMeans(n_clusters=int(len(RS_set)/2+1), random_state=1201,).fit_predict(np.array(RS_set))


cluster_label = list(kmeans_step6)
_cluster_dict = {}
for i in cluster_label:
    cnt = _cluster_dict.get(i,0)
    _cluster_dict[i] = cnt+1

RS_clster = dict()
for k,v in _cluster_dict.items():
    if v==1:
        RS_clster[k] = v

RS_clster_list = list(RS_clster.keys())


cluster_label

RS_clster_list

# Step 3. In the K-Means result from Step 2, move all the clusters that contain only one point to RS
# (outliers).
CS_data = []

RS_set_iter1 = []
for i in range(len(RS_set)):
    if cluster_label[i] not in RS_clster_list:
        CS_data.append(RS_set[i])
    else:
        RS_set_iter1.append(RS_set[i])




###Step 5. Use the K-Means result from Step 4 to generate the DS clusters (i.e., discard their points and
# generate statistics).
cluster_dict_CS = {}
for i in range(len(kmeans_step6)):
    clutser_id = kmeans_step6[i]
    clutser_list = cluster_dict_CS.get(clutser_id,[])
    clutser_list.append(RS_set[i])
    cluster_dict_CS[clutser_id] = clutser_list

cluster_CS_Stats_dict = {}
###lets generate statistics for these DS clusters 
for key,val in cluster_dict_CS.items():
    data_points = np.array(val)
    val_indexes = []
    for row in val:
        row_str = str(row)
        val_indexes.append(int(val2Index_dict[row_str]))

    sum_n = np.sum(data_points,axis=0)
    sumsq = np.sum(np.multiply(data_points,data_points),axis=0)
    centroid = sum_n/len(val)
    N = len(val)
    if N==1:
        continue
    std = np.sqrt(sumsq/N - np.square(sum_n/N))
    cluster_CS_Stats_dict[key] = {"N":N,"sum_n":sum_n,"sumsq":sumsq,"centroid":centroid,"std":std,"Members_index":val_indexes}
intermediate_result_report(0,len(RS_set_iter1))


# In[28]:


intemediate_export_dict


# In[29]:


def mahanalobis_distance(row1,row2,std):
    md_dist=0
    for dim in range(0, dimensions):
        md_dist = ((row1[dim] - row2[dim]) / std[dim]) ** 2+md_dist
    md_dist_final = np.sqrt(md_dist)
    return md_dist_final


# In[30]:


def update_DS_dict(test_point,lookupkey):
#     lookupkey = 

#     lookupkey

    row_val = test_point

    row_index = int(val2Index_dict[str(row_val)])
    
    
    dist_data = cluster_DS_Stats_dict.get(lookupkey,{})
    dist_data['Members_index'].append(row_index)

    dist_data['N'] = dist_data['N']+1

    dist_data['sum_n'] = dist_data['sum_n'] + np.array(row_val) 

    dist_data['sumsq'] = dist_data['sumsq'] +np.multiply(np.array(row_val),np.array(row_val))

    dist_data['centroid'] = dist_data['sum_n']/dist_data['N']
    #N = len(val)
    dist_data['std'] = np.sqrt(dist_data['sumsq']/dist_data['N'] - np.square(dist_data['sum_n']/dist_data['N']))


    cluster_DS_Stats_dict[lookupkey] = dist_data
#     print(cluster_DS_Stats_dict)


# In[31]:


def update_CS_dict(test_point,lookupkey):
#     lookupkey = test_point[1]

#     lookupkey

    row_val = test_point

    row_index = int(val2Index_dict[str(row_val)])

    dist_data = cluster_CS_Stats_dict.get(lookupkey,{})
    dist_data['Members_index'].append(row_index)
    dist_data['N'] = dist_data['N']+1

    dist_data['sum_n'] = dist_data['sum_n'] + np.array(row_val) 

    dist_data['sumsq'] = dist_data['sumsq'] +np.multiply(np.array(row_val),np.array(row_val))

    dist_data['centroid'] = dist_data['sum_n']/dist_data['N']
    #N = len(val)
    dist_data['std'] = np.sqrt(dist_data['sumsq']/dist_data['N'] - np.square(dist_data['sum_n']/dist_data['N']))


    cluster_CS_Stats_dict[lookupkey] = dist_data
#     print(cluster_DS_Stats_dict)


# In[32]:


def DC_CS_mapping_function(random20prct_x):
    global RS_set_iter1
    cluster_ids = []
    RS_list = []
    DS_add_list = []
    for r1 in random20prct_x:
        closest_centroid = -1
        low_mahalpbnis = 9999999997999999999999999
        for keys in cluster_DS_Stats_dict.keys():
        #     print(keys)
            centroid = list(cluster_DS_Stats_dict[keys]['centroid'])
        #     print(centroid)
            std = list(cluster_DS_Stats_dict[keys]['std'])
        #     print(std)
            key_distance = mahanalobis_distance(r1,centroid,std)
            if key_distance<threshold:
                if key_distance<low_mahalpbnis:
                    low_mahalpbnis = key_distance
                    closest_centroid = keys
    #     print(closest_centroid)
        cluster_ids.append(closest_centroid)
        if closest_centroid==-1:
            RS_list.append(r1)
        else:
            update_DS_dict(r1,closest_centroid)
    cluster_ids_CS = []
    RS_list_new = []
    CS_add_list = []
    for r1 in RS_list:
        closest_centroid = -1
        low_mahalpbnis = 9999999997999999999999999
        for keys in cluster_CS_Stats_dict.keys():
        #     print(keys)
            centroid = list(cluster_CS_Stats_dict[keys]['centroid'])
        #     print(centroid)
            std = list(cluster_CS_Stats_dict[keys]['std'])
        #     print(std)
            key_distance = mahanalobis_distance(r1,centroid,std)
            if key_distance<threshold:
                if key_distance<low_mahalpbnis:
                    low_mahalpbnis = key_distance
                    closest_centroid = keys
    #     print(closest_centroid)
        cluster_ids_CS.append(closest_centroid)
        if closest_centroid==-1:
            RS_set_iter1.append(r1)
        else:
            update_CS_dict(r1,closest_centroid)
    


# In[33]:


def kmeans_to_Cluster(kmeans_step_recur_RS,RS_set_iter1):
    return_cluster = {}
    for i in range(len(kmeans_step_recur_RS)):
        members = return_cluster.get(kmeans_step_recur_RS[i],[])
        members.append(RS_set_iter1[i])
        return_cluster[kmeans_step_recur_RS[i]] = members
    return return_cluster
        


# In[34]:


def Add_2_CS(val,key):
    data_points = np.array(val)
    val_indexes = []
    for row in val:
        row_str = str(row)
        val_indexes.append(int(val2Index_dict[row_str]))
        
    sum_n = np.sum(data_points,axis=0)
    sumsq = np.sum(np.multiply(data_points,data_points),axis=0)
    centroid = sum_n/len(val)
    N = len(val)
    std = np.sqrt(sumsq/N - np.square(sum_n/N))
    cluster_CS_Stats_dict[key] = {"N":N,"sum_n":sum_n,"sumsq":sumsq,"centroid":centroid,"std":std,"Members_index":val_indexes}


# In[35]:


## function to find nearest cluster in CS set 

def get_nearest_cluster_dict(summary1, summary2):
    global threshold,dimensions
#     cluster1_keys = summary1.keys()
#     cluster2_keys = summary2.keys()
    nearest_cluster_id_map = {}
    for key1 in summary1.keys():
        nearest_cluster_md = threshold
        nearest_clusterid = key1
        for key2 in summary2.keys():
            if key1 != key2:
                stddev1 = summary1[key1]['std']
                centroid1 = summary1[key1]['centroid']
                stddev2 = summary2[key2]['std']
                centroid2 = summary2[key2]['centroid']
                md1 = 0
                md2 = 0
                for dim in range(0, dimensions):
                    if stddev2[dim] != 0 and stddev1[dim] != 0:
                        md1 += ((centroid1[dim] - centroid2[dim]) / stddev2[dim]) ** 2
                        md2 += ((centroid2[dim] - centroid1[dim]) / stddev1[dim]) ** 2
                mahalanobis_distance = min(np.sqrt(md1), np.sqrt(md2))
                if mahalanobis_distance < nearest_cluster_md:
                    nearest_cluster_md = mahalanobis_distance
                    nearest_clusterid = key2
        nearest_cluster_id_map[key1] = nearest_clusterid
    return nearest_cluster_id_map


# In[36]:



def merge_CS_Clusters(neraest_cluster_map):
    for key,val in neraest_cluster_map.items():
        if key!=val and key in cluster_CS_Stats_dict.keys():
            if val in cluster_CS_Stats_dict.keys():
                N1 = cluster_CS_Stats_dict[key]['N']
                N2 = cluster_CS_Stats_dict[val]['N']
                sum_n1 = cluster_CS_Stats_dict[key]['sum_n']
                sum_n2 = cluster_CS_Stats_dict[val]['sum_n']
                sumsq1 = cluster_CS_Stats_dict[key]['sumsq']
                sumsq2 = cluster_CS_Stats_dict[val]['sumsq']
                ### start updating key index valuess
                newN1 = N1+N2
                new_sumN1 = sum_n1+sum_n2
                new_sumsq1 = sumsq1+sumsq2
                centroid = new_sumN1/newN1
        #N = len(val)
                std = np.sqrt(new_sumsq1/newN1 - np.square(new_sumN1/newN1))
                members1 = cluster_CS_Stats_dict[key]['Members_index'][:]
                members2 = cluster_CS_Stats_dict[val]['Members_index'][:]

                final_members = []
                for i in members1:
                    final_members.append(i)
                for i in members2:
                    final_members.append(i)

                cluster_CS_Stats_dict[key]['Members_index'] = final_members
                cluster_CS_Stats_dict[key]['std'] = std
                cluster_CS_Stats_dict[key]['centroid'] = centroid
                cluster_CS_Stats_dict[key]['sumsq'] = sumsq
                cluster_CS_Stats_dict[key]['sum_n'] = sum_n
    #             https://stackoverflow.com/questions/11277432/how-can-i-remove-a-key-from-a-python-dictionary
                del cluster_CS_Stats_dict[val]
            


# In[37]:



def merge_CS_DS_Clusters(neraest_cluster_map):
    for key,val in neraest_cluster_map.items():
        if  key in cluster_CS_Stats_dict.keys() and val in cluster_DS_Stats_dict.keys():
            N1 = cluster_CS_Stats_dict[key]['N']
            N2 = cluster_DS_Stats_dict[val]['N']
            sum_n1 = cluster_CS_Stats_dict[key]['sum_n']
            sum_n2 = cluster_DS_Stats_dict[val]['sum_n']
            sumsq1 = cluster_CS_Stats_dict[key]['sumsq']
            sumsq2 = cluster_DS_Stats_dict[val]['sumsq']
            ### start updating key index valuess
            newN1 = N1+N2
            new_sumN1 = sum_n1+sum_n2
            new_sumsq1 = sumsq1+sumsq2
            centroid = new_sumN1/newN1
    #N = len(val)
            std = np.sqrt(new_sumsq1/newN1 - np.square(new_sumN1/newN1))
            memebrs1 = cluster_CS_Stats_dict[key]['Members_index'][:]
            members2 = cluster_DS_Stats_dict[val]['Members_index'][:]

            final_members = []
            for i in memebrs1:
                final_members.append(i)
            for i in members2:
                final_members.append(i)

            cluster_DS_Stats_dict[val]['Members_index'] = final_members
            cluster_DS_Stats_dict[val]['std'] = std
            cluster_DS_Stats_dict[val]['centroid'] = centroid
            cluster_DS_Stats_dict[val]['sumsq'] = sumsq
            cluster_DS_Stats_dict[val]['sum_n'] = sum_n
#             https://stackoverflow.com/questions/11277432/how-can-i-remove-a-key-from-a-python-dictionary
            del cluster_CS_Stats_dict[key]
            


# In[38]:


# Step 7. Load another 20% of the data randomly.
### SO we take data in chunks of 20%. Initially we used 20% of the data, now we will start from that end point and pick 20% subsequent chunks of data

global dimensions
dimensions = len(input_data[0][2:])
print("Number of features in dataset {}".format(dimensions))
from numpy import sqrt
global threshold
threshold = 2*sqrt(dimensions)
print("Mahanalobis Threshold for merging two clusters {}".format(threshold))
last_round = 4
for i in range(1,5):
    epoch = i
    if epoch==last_round:
        random20prct_x = X[_20prct*epoch:]
    else:
        random20prct_x = X[_20prct*epoch:_20prct*(epoch+1)]
    print("Lets try to map new datapoints to DS/CS Cluster")
    DC_CS_mapping_function(random20prct_x)
    kmeans_step_recur_RS = KMeans(n_clusters=int(len(RS_set_iter1)/2+1), random_state=1201,).fit_predict(np.array(RS_set_iter1))
    new_clutsers = kmeans_to_Cluster(kmeans_step_recur_RS,RS_set_iter1)
    ## Identify RS Points and store them in RS_set_iter1
    RS_set_iter1 = [] 
    for k,v in new_clutsers.items():
    #     print(k,len(v))
        if len(v)==1:
            RS_set_iter1.append(v[0])
        
    ### Adding new CS clusters to Existing compression set 

    for k,v in new_clutsers.items():
    #     print(k,len(v))
        if len(v)!=1:
    #         insert_key = 0
            if k in cluster_CS_Stats_dict.keys():
                insert_key = max(list(cluster_CS_Stats_dict.keys()))+1
#                 while k in cluster_CS_Stats_dict:
#                     k+=1
#                 insert_key = k
            else:
                insert_key = k
            Add_2_CS(v,insert_key)
            
            ## Step 12 : Merge CS clusters which have MD less than 2 * sqrt(d)

#     CS_keys = list(cluster_CS_Stats_dict.keys())
    # closest_cluster_map = get_nearest_cluster_dict(compression_set, compression_set)
    neraest_cluster_map = get_nearest_cluster_dict(cluster_CS_Stats_dict,cluster_CS_Stats_dict)
    merge_CS_Clusters(neraest_cluster_map)
    
    ### what if this is our last round, we need to merge 

    if epoch == last_round:
        cs_ds_merging_map = get_nearest_cluster_dict(cluster_CS_Stats_dict, cluster_CS_Stats_dict)
        merge_CS_DS_Clusters(cs_ds_merging_map)
    rs_cnt = len(RS_set_iter1)
    intermediate_result_report(epoch,rs_cnt)
    

    # At each run, including the initialization step, you need to count and output the number of the discard
    # points, the number of the clusters in the CS, the number of the compression points, and the number of
    # the points in the retained set.
    


            
            

    


# In[39]:


member_id_cluster_info = []
for key,val in cluster_DS_Stats_dict.items():
    for member in set(val['Members_index']):
        member_id_cluster_info.append((member,key))
        
    


# In[40]:


for key,val in cluster_CS_Stats_dict.items():
    for member in set(val['Members_index']):
        member_id_cluster_info.append((member,-1))


# In[41]:


if len(RS_set_iter1)>0:
    for w in RS_set_iter1:
        index = int(val2Index_dict[str(w)])
        print(index)
        member_id_cluster_info.append((index,-1))
        


# In[42]:


export_list = sorted(member_id_cluster_info,key=lambda x : x[0])


# In[43]:


with open(output_file_path,'w+') as fout:
    
    for key,val in intemediate_export_dict.items():
        fout.write(val)
    fout.write("\nThe clustering results: ")
    for row  in export_list:
# clustered_data.append([point, point_clusterid_map[point]])
        fout.write("\n" + str(int(row[0])) + "," + str(row[1]))

fout.close()

print("Task Completed. File exported successfully")

end = time.time()

print("Time taken by process : {}".format(end-start))

    


# In[ ]:




