import numpy

import glob

import re

import sys

import os

import json



stop_words=['other', 'first',
            'been', 'their', 'some', 'more', 'what', 'got', 
            'also', 'here', 'only', 'or', 'which', 'by', 'could', 'even',
            'did', 'after', 'about', 'will', 'just', 'again', 'get',
            'if', 'up', 'us', 'out', 'an', 
            'one', 'are', 'me', 'so', 'all',
            'when', 'would', 'be', 'from', 'as', 'you', 'there', 'have',
            'stay', 'very', 'but', 'our', 'they', 'on', 'had', 'with', 
            'this', 'were', 'is', 'that', 'at', 'my', 'it', 'for',
            'hotel', 'we', 'of', 'in', 'was', 'i', 'a', 'to', 'and', 'the', 'hotels','went',
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




file = open('nbmodel.txt', 'r')



# In[19]:
paths = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))
print("test data ....")

print(len(paths))

      
all_files = paths

# In[20]:




classVsWordProbablities = json.loads(file.read())
result={}
file = open("nboutput.txt","w") 


import math

def predict_class(model, content):
    words = content.split()
    max = float('-inf')
    class_res = 0
    class_prob={}
    for key,value in model.items():
        probablity = 0
        for word in words:
            val = model.get(key).get(word, None)
            if val != None:
                val = math.log(val)
                probablity += val
        class_prob[key] = probablity * 1/2
    for key,value in class_prob.items():
        if value > max :
            max = value
            class_res = key
    return class_res



pos_neg_probs = classVsWordProbablities['probs_neg_pos']

probs_true_decept = classVsWordProbablities['probs_true_decept']

print(classVsWordProbablities.keys())

# In[23]:

file = open("nboutput.txt","w") 
for l in all_files[:]:
    with open(l) as f:
        comment = open(l).read()
#     print(comment)
    test_comments = normalizeData(comment)
    val = predict_class(pos_neg_probs, test_comments)
    #print(val)
    if val=='0':
        pred1 = 'negative'
    else:
        pred1 = 'positive'
    val = predict_class(probs_true_decept, test_comments)
    if val=='0':
        pred2 = 'deceptive'
    else:
        pred2 = 'truthful'

    file.write(pred2+" "+pred1+" "+ l)
    #print(pred2+" "+pred1+" "+ l)
    file.write("\n")
file.close()

print("Scoring comepleted")
