import json,sys,numpy as np

def viterbi_path(prior, transmat, obslik):
    scaled=True
    ret_loglik=False
    num_hid = obslik.shape[0] 
    num_obs = obslik.shape[1]

    trellis_prob = np.zeros((num_hid,num_obs))
    # trellis_state[i,t] := best predecessor state given that we ended up in state i at t
    trellis_state = np.zeros((num_hid,num_obs), dtype=int) # int because its elements will be used as indicies
    path = np.zeros(num_obs, dtype=int) # int because its elements will be used as indicies

    trellis_prob[:,0] = prior * obslik[:,0] # element-wise mult
    if scaled:
        scale = np.ones(num_obs) # only instantiated if necessary to save memory
        scale[0] = 1.0 /(0.00001+np.sum(trellis_prob[:,0]))
        trellis_prob[:,0] = trellis_prob[:,0]*scale[0]

    trellis_state[:,0] = 0 # arbitrary value since t == 0 has no predecessor
    for t in range(1, num_obs):
        for j in range(num_hid):
            trans_probs = trellis_prob[:,t-1] * transmat[:,j] # element-wise mult
            trellis_state[j,t] = trans_probs.argmax()
            trellis_prob[j,t] = trans_probs[trellis_state[j,t]] # max of trans_probs
            trellis_prob[j,t] = trellis_prob[j,t]* obslik[j,t]
        if scaled:
            scale[t] = 1.0 / (np.sum(trellis_prob[:,t])+0.0000000001)
            trellis_prob[:,t] =trellis_prob[:,t]* scale[t]

    path[-1] = trellis_prob[:,-1].argmax()
    for t in range(num_obs-2, -1, -1):
        path[t] = trellis_state[(path[t+1]), t+1]

    if not ret_loglik:
        return path
    else:
        if scaled:
            loglik = -np.sum(np.log(scale))
        else:
            p = trellis_prob[path[-1],-1]
            loglik = np.log(p)
        return path, loglik

# ## Italian Language HMM

# In[614]:




json_file_path = 'hmmmodel.txt'

with open(json_file_path, 'r') as j:
     model_dict = json.loads(j.read())

# pathRead = 

# model_dict = json.loads(pathRead)

tag_string_index=  model_dict["tag_string_index"]

tag_index_string = model_dict["tag_index_string"] 

Tag_prior_prob = model_dict["Tag_prior_prob"] 

State_Transition_matrix = model_dict["State_Transition_matrix"] 

Tag_set_italian = model_dict["Tag_set_italian"] 


Words_set_italian = model_dict["Words_set_italian"] 

words_index_string_tmp = model_dict["words_index_string"] 

Tag_observation_matrix = model_dict["Tag_observation_matrix"] 

tag_index_string_tmp= model_dict["tag_index_string"] 


words_string_index = model_dict["words_string_index"] 


Tag_frequency = model_dict["Tag_frequency"] 


words_string_index.get("UNK_WORD_DATA")



### Added this 



###### Addtion over ######


pi = np.zeros(len(Tag_prior_prob))


# In[629]:


for k,v in tag_string_index.items():
    pi[v] = Tag_prior_prob[k]
    





A = np.zeros(len(Tag_prior_prob)*len(Tag_prior_prob)).reshape(len(Tag_prior_prob),len(Tag_prior_prob))




A.shape

tag_index_string = {}

for k,v in tag_index_string_tmp.items():
    tag_index_string[int(k)] = v

words_index_string = {}

for k,v in words_index_string_tmp.items():
    words_index_string[int(k)] = v


for i in range(A.shape[0]):
    row = tag_index_string[i]
    for j in range(A.shape[0]):
        col = tag_index_string[j]
        prob = State_Transition_matrix[row][col]
        if prob==0:
            prob=0.0001
        
        A[i][j] = prob


B = np.zeros(len(Tag_set_italian)*len(Words_set_italian)).reshape(len(Tag_set_italian),len(Words_set_italian))

for i in range(B.shape[0]):
    row = tag_index_string[i]
    for j in range(B.shape[1]):
        col = words_index_string[j]
        prob = Tag_observation_matrix[row][col]
        
        B[i][j] = prob





def update(B):
    if "UNK_WORD_DATA" in words_string_index:
        return B
    else:
        print("Update happening")
    
        words_string_index["UNK_WORD_DATA"] = len(words_string_index)

        words_index_string[len(words_string_index)] = "UNK_WORD_DATA"

        weight_vector_tags = np.zeros(len(Tag_frequency)).reshape(len(Tag_frequency),1)

        overall_count = 0
        for k,v in Tag_frequency.items():
            overall_count+=v


        Tag_frequency_percentage = {}
        for k,v in Tag_frequency.items():
            index = tag_string_index.get(k)
            Tag_frequency_percentage[index] = v/overall_count

        for k,v in Tag_frequency_percentage.items():
            if v<0.01:
                Tag_frequency_percentage[k]= 0
        for i in range(len(weight_vector_tags)):

            weight_vector_tags[i] = Tag_frequency_percentage.get(i)

        return np.hstack((B,weight_vector_tags))

    

B_tmp = update(B)




    







pathTest = sys.argv[1]

fb = open(pathTest, encoding = 'UTF-8')



testing_data = fb.readlines()
writeFilePath = 'hmmoutput.txt'
writeFile = open(writeFilePath, mode = 'w', encoding = 'UTF-8')
for sentence in testing_data:
    obs = []
    tags_index = []
    for words in sentence.split():
        word = words
        if word not in words_string_index:
            word='UNK_WORD_DATA'
        obs.append(words_string_index[word])
    prior = pi
    transmat = A
    emmat = B_tmp
    observations = np.array(obs, dtype=int)
    obslik = np.array([emmat[:,z] for z in observations]).T
    prediction= viterbi_path(prior, transmat, obslik)
    
    tags = [tag_index_string.get(w) for w in prediction]
    
    write_vector =[w for w in sentence.split()]
    
    write_string = ''
    
    for w in list(zip(write_vector,tags)):
        write_string+=str(w[0])+"/"+str(w[1])+" "
    
    writeFile.write(write_string.rstrip())
    writeFile.write("\n")
    
