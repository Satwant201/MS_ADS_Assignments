#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob,numpy as np, sys,os,json


# In[523]:






training_data = []

path = sys.argv[1]


with open(path, encoding = 'UTF-8') as fb:
    training_data= fb.readlines()



Tag_set_italian = set()
Words_set_italian = set()
tag_counter = 0
Tag_frequency = {}
State_observation_freq = {}
Tag_prior_cnt = {}

Number_training_count = 0


for sentence in training_data:
    Number_training_count+=1
    for index,word in enumerate(sentence.split()):
        tag = word.split("/")[-1]
        word = word.split("/")[0]
        if index==0:
            cnt = Tag_prior_cnt.get(tag,0)
            Tag_prior_cnt[tag] = cnt+1

        tag_dict = State_observation_freq.get(tag,{})
        
        count_word = tag_dict.get(word,0)
        tag_dict[word] = count_word+1
        State_observation_freq[tag] = tag_dict
        
        count = Tag_frequency.get(tag,0)
        Tag_frequency[tag] = count+1
        Words_set_italian.add(word)
#         print(word.split("/"))
        Tag_set_italian.add(tag)
    
        tag_counter+=1
#         break
#     break


# In[616]:


Number_training_count


# ### Prior Probability of All Tags generated using add 1 smoothening

# In[617]:


Tag_prior_prob = {}

for tag in Tag_set_italian:
    cnt = Tag_prior_cnt.get(tag,0)
    Tag_prior_prob[tag] = (cnt)/Number_training_count


# ## State Observation Transition Matrix Generated

# In[618]:


# for k,v in Tag_frequency.items():
#     inner_dict = State_observation_freq.get(k,{})
#     output_dict1 = {}
#     for k1,v1 in inner_dict.items():
#         output_dict1[k1] = v1/v
    
#     State_observation_freq[k] = output_dict1


# In[619]:


Tag_obs_matrix = []
Tag_observation_matrix = {}
for tag in Tag_set_italian:
    dict1 = {}
    for word in Words_set_italian:
        cnt = State_observation_freq.get(tag,{}).get(word,0)
        denom = Tag_frequency.get(tag,0)
        dict1[word] = cnt/denom
        Tag_obs_matrix.append((tag,word,cnt/denom))
    
    Tag_observation_matrix[tag] = dict1


# ## Lets Generate State Transtion Matrix
# 
# First we need to make bigrams
# 
# Then generate freq for each bigram occurance
# 
# Then, we generate an exhaustive pair wise count of each tag pair, note that majority would be 0 as such tags never occur together
# 

# In[620]:


## Tag Transition Matrix 
tag_transition_matrix ={}
tag_transition_counter = {}

for sentence in training_data:
    tokens = sentence+" /EOS"
    tokens = tokens.split()

    vector_length = len(tokens)
    for i in range(len(tokens)):
        tag_i = tokens[i].split("/")[-1]
        
        if i!=vector_length-1:
            tag_i_1 = tokens[i+1].split("/")[-1]
            matrix = tag_transition_matrix.get(tag_i,{})
            count = matrix.get(tag_i_1,0)
            matrix[tag_i_1] = count+1
            tag_transition_matrix[tag_i] = matrix
            cnt = tag_transition_counter.get(tag_i,0)
            tag_transition_counter[tag_i]=cnt+1
        else:
            continue
        


# In[621]:


Tag1_Tag2_cnt = {}
for tag1 in Tag_set_italian:
    tag1_mat = tag_transition_matrix.get(tag1,{})
    for tag2 in Tag_set_italian:
        cnt = tag1_mat.get(tag2,0)
        
        
        tag1_mat[tag2] = cnt
    
    Tag1_Tag2_cnt[tag1] =  tag1_mat   
    


# In[622]:



########## Comment line 167 to 179
#State_Transition_matrix = {}


# In[623]:

'''
for k,v in tag_transition_counter.items():
    transitions1 = Tag1_Tag2_cnt.get(k)
    
    for k1,v1 in transitions1.items():
        transitions1[k1] = v1/v
    
    State_Transition_matrix[k] = transitions1
'''

State_Transition_matrix = {}

import math
# In[623]:
logFactor = 2*len(tag_transition_counter)
# for currTag in tagGivenTag:
#     for nextTag in tagGivenTag[currTag]:
#         tagGivenTag[currTag][nextTag] = (math.log(tagGivenTag[currTag][nextTag]) + logFactor) - math.log(tagCount[currTag])


for k,v in tag_transition_counter.items():
    transitions1 = Tag1_Tag2_cnt.get(k)
    for k1,v1 in transitions1.items():
        
        transitions1[k1] = v1/v

    
    State_Transition_matrix[k] = transitions1



# ## Lets Genrate Pi, A, B after converting strings to numeric indexes

# #### Create index dictionaries

# ### Tags

# In[624]:


tag_string_index = {}
tag_index_string = {}
tag_list = list(Tag_set_italian)
for i in range(len(tag_list)):
    tag_string_index[tag_list[i]] = i
    tag_index_string[i] = tag_list[i]
    


# ### Words

# In[659]:


words_string_index = {}
words_index_string = {}
words_list = list(Words_set_italian)
for i in range(len(words_list)):
    words_string_index[words_list[i]] = i
    words_index_string[i] = words_list[i]
    


# ### Lets Generate Priors Numpy Array

# In[626]:


print("Number of Unique Tags in Data : {}".format(len(Tag_prior_prob)))


# In[627]:


print("Number of Items in Tag Lookup Dictionary {}".format(len(tag_string_index)))


# In[628]:


pi = np.zeros(len(Tag_prior_prob))


# In[629]:


for k,v in tag_string_index.items():
    pi[v] = Tag_prior_prob[k]
    
    


# ## Lets Generate A

# In[630]:




A = np.zeros(len(Tag_prior_prob)*len(Tag_prior_prob)).reshape(len(Tag_prior_prob),len(Tag_prior_prob))


# In[631]:


for i in range(A.shape[0]):
    row = tag_index_string[i]
    for j in range(A.shape[0]):
        col = tag_index_string[j]
        prob = State_Transition_matrix[row][col]
        
        A[i][j] = prob


# ## Lets Generate B

# In[650]:


B = np.zeros(len(Tag_set_italian)*len(Words_set_italian)).reshape(len(Tag_set_italian),len(Words_set_italian))


# In[651]:





for i in range(B.shape[0]):
    row = tag_index_string[i]
    for j in range(B.shape[1]):
        col = words_index_string[j]
        prob = Tag_observation_matrix[row][col]
        
        B[i][j] = prob


model_dict = {}













model_dict["tag_string_index"] = tag_string_index

model_dict["tag_index_string"] = tag_index_string

model_dict["Tag_prior_prob"] = Tag_prior_prob

model_dict["State_Transition_matrix"] = State_Transition_matrix

model_dict["Tag_set_italian"] = list(Tag_set_italian)


model_dict["Words_set_italian"] = list(Words_set_italian)

model_dict["words_index_string"] = words_index_string

model_dict["Tag_observation_matrix"] = Tag_observation_matrix

model_dict["tag_index_string"] = tag_index_string

model_dict["words_string_index"] = words_string_index

model_dict["Tag_frequency"] = Tag_frequency


Tag_frequency
words_string_index

writeFilePath = 'hmmmodel.txt'
writeFile = open(writeFilePath, mode = 'w', encoding = 'UTF-8')
writeFile.write(json.dumps(model_dict))
