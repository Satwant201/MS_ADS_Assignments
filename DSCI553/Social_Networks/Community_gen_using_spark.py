import time
import os
import sys

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row


from graphframes import GraphFrame


os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12")




conf = SparkConf().setAppName("DSCI553_SP22").setMaster('local[*]')

sc = SparkContext(conf=conf)

sc.setLogLevel("ERROR")

sqlContext = SQLContext(sc)

start_time = time.time()
filter_threshold = int(sys.argv[1])
input_file = sys.argv[2]
output_file = sys.argv[3]

input_data_tmp = sc.textFile(input_file)

# >>> input_data_raw.take(2)
# [['user_id', 'business_id'], ['39FT2Ui8KUXwmUt6hnwy-g', 'RJSFI7mxGnkIIKiJCufLkg']]

header = input_data_tmp.take(1)[0]
input_data_raw = input_data_tmp.filter( lambda x : x!=header).map(lambda x: x.split(",")).map(lambda x: (x[0],x[1]))
# >>> input_data_raw.take(1)
# [('39FT2Ui8KUXwmUt6hnwy-g', 'RJSFI7mxGnkIIKiJCufLkg')]

'''
Currenlty each row represnts user, business pair. we need to first identify valid user-user pairs and transform dataset as user1,user2 format where valid pairs
Have minimum common business count > filter passed as param

Self join
'''
def extact_valid_pairs(x):
    if len(x[0][1])>0:
        if len(x[1][1])>0:
            b1 = x[0][1]
            b2 = x[1][1]
            threh = len(b1.intersection(b2))
            if threh>=filter_threshold:
                return 1
            else:
                return 0


user_business_set = input_data_raw.groupByKey().mapValues(set)

input_data =user_business_set.cartesian(user_business_set).\
                                        filter(lambda x : x[0][0]!=x[1][0]).\
                                        filter(extact_valid_pairs).\
                                        map(lambda x: tuple(sorted([x[0][0],x[1][0]]))).distinct()


            
            


def get_all_vertices(user1_rdd,user2_rdd):
    return sc.union([user1_rdd, user2_rdd]).distinct().map(lambda x: Row(x))


user1_rdd = input_data.map(lambda user: user[0])

user2_rdd = input_data.map(lambda user: user[1])

user_vertices_rdd = get_all_vertices(user1_rdd,user2_rdd)

def get_all_edges(node1_edge,node2_edge):
    return sc.union([node1_edge, node2_edge]).distinct()

node2_edge = input_data.map(lambda x1: (x1[1], x1[0]))

node1_edge = input_data.map(lambda x1: (x1[0], x1[1]))



'''
@References:

1) https://docs.databricks.com/spark/latest/graph-analysis/graphframes/graph-analysis-tutorial.html
2) https://docs.databricks.com/_static/notebooks/graphframes-user-guide-py.html
3) Hw4 assignment pdf
4) Lecture notes
5) https://docs.microsoft.com/en-us/azure/databricks/spark/latest/graph-analysis/graphframes/graph-analysis-tutorial

'''

edges_rdd = get_all_edges(node1_edge,node2_edge)

edges = sqlContext.createDataFrame(edges_rdd, ["src", "dst"])


vertices = sqlContext.createDataFrame(user_vertices_rdd, ["id"])


graph_datset = GraphFrame(vertices, edges)
grahp_output_lpa = graph_datset.labelPropagation(maxIter=5)
output_data = grahp_output_lpa.rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(lambda x: sorted(list(x))) \
    .sortBy(lambda x: (len(x[1]), x[1])).map(lambda x: tuple(x[1])).collect()



print("Entering write part fot Task1")


with open(output_file, 'w') as fout:
    for row in output_data:
        row_str = ''
        for graph_nd in row:
            row_str = row_str + "'" + str(graph_nd) + "', "
        
        row_str = row_str[:-2] ## required to remove trailing comma and space
        fout.write(row_str+"\n")
fout.close()


print("Task 1 completed")


print('Duration: {}'.format(time.time() - start_time))
