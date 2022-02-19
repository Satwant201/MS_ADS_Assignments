#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy

import glob

import re

import sys

import os

stop_words=['of', 'in', 'was', 'i', 'a', 'to', 'and', 'the',
            'when', 'would', 'be', 'from', 'as', 'you', 'there', 'have','other', 'first',
            'been', 'their', 'some', 'more', 'what', 'got', 
            'also', 'here', 'only', 'or', 'which', 'by', 'could', 'even',
            'did', 'after', 'about', 'will', 'just', 'again', 'get',
            'if', 'up', 'us', 'out', 'an', 
            'one', 'are', 'me', 'so', 'all',
            'stay', 'very', 'but', 'our', 'they', 'on', 'had', 'with', 
            'this', 'were', 'is', 'that', 'at', 'my', 'it', 'for',
            'hotel', 'we', 'hotels','went',
           "seemed"]

lookup_dict = {'less': 'less',
 'pictures': 'picture',
 'stories': 'story',
 'beds': 'bed',
 'memories': 'memory',
 'doormen': 'doorman'
 }    


def bigram(content):
    tokens = content.split()
    
    count = ''
    for i in range(len(tokens)):
    
        try:
            count+=tokens[i]+"_"+tokens[i+1]+" "
        except:
            pass
    return count.rstrip()



def normalizeData(string):
    new_string = re.sub(r"[^a-zA-Z ]"," ",string.lower().rstrip().lstrip())
    token11 = [w for w in new_string.split() if w not in stop_words]
    token1 = []
    for w in token11:
        if len(w)>2:
            if w[-1]=='s':
                if w[-2]=='e':
                    token1.append(w[:-2])
                else:
                    token1.append(w[:-1])
            else :
                token1.append(w)
        
    tk1 = []
    for i in token1:
        val = lookup_dict.get(i,None)
        if val!=None:
            tk1.append(val)
        else:
            tk1.append(i)        
    tokens = " ".join(tk1)
    return tokens


# In[2]:

training_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))
# training_files = glob.glob("demo/*/*/*/*.txt")


# In[3]:


### Data loading and pre processing


# In[4]:


comments = []

observations =0

observation_per_class = {}

words_per_class_pos_neg = {}


words_per_class_true_decept = {}

class_word_frqueny_pos_neg = {}

class_word_frqueny_true_decept = {}

vocabulary_set = set()

class_prior_probability ={}

class_word_probability_pos_neg = {}

class_word_probability_true_decept = {}


class_comment_pos_neg = []

class_comment_true_deceptive = []

for l in training_files:
    observations+=1
    with open(l) as f:
        comment = open(l).read()
        comments.append(normalizeData(comment))
        if l.find("positive")>-1:
            class_comment_pos_neg.append(1)
        else:
            class_comment_pos_neg.append(0)
        if l.find("truthful")>-1:
            class_comment_true_deceptive.append(1)
        else:
            class_comment_true_deceptive.append(0)
            
print("Total Number of Observations {}".format(observations))       
        


# In[5]:


import numpy as np


# In[6]:


### calculate class_frequency and calculating priors

class_prior_probability["positive"] = sum(class_comment_pos_neg)/np.float64(observations)
class_prior_probability["negative"] = 1-(sum(class_comment_pos_neg)/observations)
class_prior_probability["truthful"] = sum(class_comment_true_deceptive)/observations
class_prior_probability["deceptive"] = 1-(sum(class_comment_true_deceptive)/observations)


print(class_prior_probability)


# In[7]:


## calculating word frequency in each class

for k in range(len(comments)):
    class_comment_pn = class_comment_pos_neg[k]
    class_comment_td = class_comment_true_deceptive[k]
    
    actual_dict1 = class_word_frqueny_pos_neg.get(class_comment_pn,{})
    actual_dict2 = class_word_frqueny_true_decept.get(class_comment_td,{})
    
    words = comments[k].split()
    for w in words:
        count = actual_dict1.get(w,0)
        actual_dict1[w] = count+1
        class_word_frqueny_pos_neg[class_comment_pn]=actual_dict1

        count = actual_dict2.get(w,0)
        actual_dict2[w] = count+1
        class_word_frqueny_true_decept[class_comment_td]=actual_dict2


# In[8]:


a = 9


# In[9]:


## calculating words per class

    
for key,value in class_word_frqueny_pos_neg.items():
    count = 0
    for key1,val1 in value.items():
        vocabulary_set.add(key1)
        count+=val1
    words_per_class_pos_neg[key]= count

print("Positive Negative Classes")

print(words_per_class_pos_neg)

print("*"*50)

words_per_class_true_decept = {}
for key,value in class_word_frqueny_true_decept.items():
    count = 0
    for key1,val1 in value.items():
        vocabulary_set.add(key1)
        count+=val1
    words_per_class_true_decept[key]= count

print("True Deceptive Classes")
    
print(words_per_class_true_decept)
    
print("*"*50)

print("Vocabulary Count {}".format(len(vocabulary_set)))


# In[10]:


wordVsOverallCount = {}


# In[11]:


for w in " ".join(comments).split():
    count = wordVsOverallCount.get(w.lower(),0)
    wordVsOverallCount[w.lower()] = count+1
    
    


# In[12]:


sorted_by_value = sorted(wordVsOverallCount.items(), key=lambda x: x[1])


# In[13]:



stop_words=['other', 'really', 'first',
            'been', 'their', 'some', 'more', 'what', 'got', 
            'also', 'here', 'only', 'or', 'which', 'by', 'could', 'even',
            'did', 'after', 'about', 'will', 'just', 'again', 'get',
            'if', 'up', 'us', 'out', 'an', 
            'one', 'are', 'me', 'so', 'all',
            'when', 'would', 'be', 'from', 'as', 'you', 'there', 'have',
            'stay', 'very', 'but', 'our', 'they', 'on', 'had', 'with', 
            'this', 'were', 'is', 'that', 'at', 'my', 'it', 'for',
            'hotel', 'we', 'of', 'in', 'was', 'i', 'a', 'to', 'and', 'the', 'hotels','went']
for key,value in sorted_by_value:
    if value < 2:
        stop_words.append(key)
vocab_set_final = [word for word in vocabulary_set if word not in stop_words]
alpha = 0.6
# In[14]:

print(len(vocab_set_final))


# In[ ]:





# In[15]:


for key,val in class_word_frqueny_pos_neg.items():
    #freq = words_per_class_pos_neg.get(key)
        wordVsProb = class_word_probability_pos_neg.get(key,{})
        for vocabword in vocab_set_final:
                valueIn = val.get(vocabword, 0)
                probablity = (valueIn + alpha) / (words_per_class_pos_neg[key] + alpha * len(vocab_set_final))
                wordVsProb[vocabword] = probablity
        class_word_probability_pos_neg[key] = wordVsProb


# In[16]:



for key,val in class_word_frqueny_true_decept.items():
        wordVsProb = class_word_probability_true_decept.get(key,{})
        for vocabword in vocab_set_final:
                valueIn = val.get(vocabword, 0)
                probablity = (valueIn + alpha) / (words_per_class_true_decept[key] + alpha * len(vocab_set_final))
                wordVsProb[vocabword] = probablity
        class_word_probability_true_decept[key] = wordVsProb







# In[17]:


model = {}
model["priors"] = class_prior_probability
model['probs_neg_pos'] = class_word_probability_pos_neg
model['probs_true_decept'] = class_word_probability_true_decept


# In[18]:


import json
f = open("nbmodel.txt","w")
f.write(json.dumps(model))
f.close()

print("Training Completed")






# In[ ]:





# In[ ]:





# In[21]:


# In[ ]:





# In[ ]:





# In[ ]:




