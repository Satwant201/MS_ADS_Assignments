import binascii
# int(binascii.hexlify(s.encode('utf8')),16)

from blackbox import BlackBox

import time

import math 

start = time.time()

import sys

path = sys.argv[1]

stream_size = int(sys.argv[2])

num_of_asks = int(sys.argv[3])

output_filename = sys.argv[4]



global num_functions
num_functions=10
global m 

m = 100001159
def get_func(num_functions):
    func_list= []
    p= [400069,600091,450029,50053,670049,820067,840023,890053,45763,890119]
    a = [16, 175, 83, 54, 134,159,345,231,12,897]
    b = [1770, 345, 722, 1072, 655, 1995, 607, 63, 792, 1796]
    for i in range(num_functions):
        func_list.append((p[i],a[i],b[i]))
    return func_list

def myhashs(user_id_int):
    global m
    global num_functions
    result = []
    function_list = get_func(num_functions)
#  ((ax + b) % p) % m
    for func in function_list:
        hash_output = ((func[1]*user_id_int+func[2])%func[0])%m
        result.append(hash_output)
    return result

#  ((ax + b) % p) % m


def trailing(s):
    return len(s) - len(s.rstrip('0'))

global train_zero_cnt
train_zero_cnt= [0]*num_functions

def FMA(stream_element):
    global train_zero_cnt
    user_id_int = int(binascii.hexlify(stream_element.encode('utf8')),16)
    user_hash_list = myhashs(user_id_int)
#     print(user_id_int)
    for index,hash_val in enumerate(user_hash_list):
#         has_val_out = hash_val(user_id_int)
#         print(hash_val)
        binary_represent = bin(hash_val)
#         print(binary_represent)
        cnt = trailing(binary_represent)
        if train_zero_cnt[index]<=cnt:
            train_zero_cnt[index]=cnt



# python task1.py <input_filename> stream_size num_of_asks <output_filename>

# path = path = "/Users/satwant/Documents/Spring22/DSCI 553/Assignments/Assignment5/Data/users.txt"

# stream_size = 300

# 
# num_of_asks = 30

# output_filename = "/Users/satwant/Documents/Spring22/DSCI 553/Assignments/Assignment5/Data/task2_output.txt"

with open(output_filename,"w") as fout:
    fout.write("Time,Ground Truth,Estimation\n")
    actual_cnt_log = 0
    predicted_cnt_log = 0

    for iteration in range(num_of_asks):
        train_zero_cnt= [0]*num_functions
        bx = BlackBox()
        data_stream = bx.ask(path,stream_size)
        actual=len(set(data_stream))
        actual_cnt_log+=actual
        predicted = 0
        for used_id in data_stream:
            FMA(used_id)
        mean_distinct = sum(train_zero_cnt)/num_functions
        predicted = int(math.pow(2,math.floor(mean_distinct)))
        predicted_cnt_log+=predicted

                # print(fp_rate)
        write_val = str(iteration)+","+str(actual)+","+str(predicted)+"\n"
        fout.write(write_val)
        print(write_val)
    print("Overall Ratio is {}".format(predicted_cnt_log/actual_cnt_log))

fout.close()


end = time.time()

print("Duration : {}".format(end-start))
  
    
        
    







