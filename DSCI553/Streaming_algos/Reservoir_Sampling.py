from blackbox import BlackBox
import random
import sys
import time


def reservoir_sampling(stream_users, ask):
    global reservoir, seqnum
    if ask == 0:
        for seqnum in range(stream_size):
            reservoir[seqnum] = stream_users[seqnum]
        seqnum += 1
    else:
        for user in stream_users:
            keep_probbality = stream_size / seqnum
            # random.randint() % q < p will simulate a prob of p / q
            # we need to keep user with prob 100/n, where n ranges from 1 to ask*100, i.e seqnum + 1
            if random.random() < stream_size/float(seqnum+1):
                x = replace = random.randint(0,len(reservoir)-1)  # chossing the location in list
                reservoir[x] = user
            seqnum += 1
    f.write(str(seqnum) + ',' + str(reservoir[0]) + ',' + str(reservoir[20]) + ',' + str(reservoir[40]) + ',' + str(
        reservoir[60]) + ',' + str(reservoir[80]) + '\n')


if __name__ == "__main__":
    # time python3 task3.py $ASNLIB/publicdata/users.txt 100 30 task3.csv
    start_time = time.time()
#     path = "/Users/satwant/Documents/Spring22/DSCI 553/Assignments/Assignment5/Data/"
#     input_file = path+"users.txt"
#     stream_size = 100
#     num_of_asks = 30
#     output_file = path+"task3_out.txt"

    input_file = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file = sys.argv[4]

    random.seed(553)
    reservoir = [0] * stream_size
    seqnum = 0

    f = open(output_file, "w")
    f.write("seqnum,0_id,20_id,40_id,60_id,80_id\n")

    bx = BlackBox()
    for ask in range(num_of_asks):
        stream_users = bx.ask(input_file, stream_size)
        reservoir_sampling(stream_users, ask)
    f.close()
    print("Duration : ", time.time() - start_time)
