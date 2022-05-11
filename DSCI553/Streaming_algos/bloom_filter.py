import binascii
# int(binascii.hexlify(s.encode('utf8')),16)

from blackbox import BlackBox

import time


start = time.time()

import sys


global num_functions
num_functions=10
global m 


m = 69997
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

global lookup_array

lookup_array = [0]*m

global user_seen_set

user_seen_set = set()


def bloomfilter(stream_element):
    global lookup_array
    global user_seen_set
    y_pred = 0
    y_true= 0
    user_id_int = int(binascii.hexlify(stream_element.encode('utf8')),16)
    bit_maps_set = 0
    user_hash_list = myhashs(user_id_int)
    for index in user_hash_list:
        if lookup_array[index]==1:
            bit_maps_set+=1
        else:
            lookup_array[index]=1
    if stream_element not in  user_seen_set:
        y_true=0
        if bit_maps_set == num_functions:
            y_pred = 1
        else:
            y_pred=0
    if stream_element in  user_seen_set:
        y_true=1
        if bit_maps_set == num_functions:
            y_pred = 1
        else:
            y_pred=0
    user_seen_set.add(stream_element)
    return y_true,y_pred



path = sys.argv[1]

stream_size = int(sys.argv[2])

num_of_asks = int(sys.argv[3])

output_filename = sys.argv[4]

with open(output_filename,"w") as fout:
    fout.write("Time,FPR\n")

    for iteration in range(num_of_asks):
        bx = BlackBox()
        data_stream = bx.ask(path,stream_size)
        fpr = 0
        tpr = 0
        for used_id in data_stream:
    #         user_id_int = int(binascii.hexlify(used_id.encode('utf8')),16)
            y_true,y_pred = bloomfilter(used_id)
            if y_true==0 and y_pred==1:
                fpr+=1
            if y_true==0 and y_pred==0:
                tpr+=1
        fp_rate = fpr/(fpr+tpr)
        # print(fp_rate)
        write_val = str(iteration)+","+str(fp_rate)+"\n"
        print(write_val)
        fout.write(write_val)

fout.close()


end = time.time()

print("Duration : {}".format(end-start))
    
    
# python task1.py <input_filename> stream_size num_of_asks <output_filename>

    
        
    







