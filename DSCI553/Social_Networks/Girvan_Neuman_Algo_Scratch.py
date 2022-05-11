
'''
@References:

1) https://docs.databricks.com/spark/latest/graph-analysis/graphframes/graph-analysis-tutorial.html
2) https://docs.databricks.com/_static/notebooks/graphframes-user-guide-py.html
3) Hw4 assignment pdf
4) Lecture notes
5) https://docs.microsoft.com/en-us/azure/databricks/spark/latest/graph-analysis/graphframes/graph-analysis-tutorial

'''



from collections import defaultdict
from operator import add
# from pyspark import SparkContext
import random



import copy
import time
import sys

from operator import add

filter_threshold = int(sys.argv[1])
input_file = sys.argv[2]
betweenness_output_file = sys.argv[3]
community_output_file = sys.argv[4]



from pyspark import SparkContext


sc = SparkContext()
sc.setLogLevel("ERROR")


# filter_threshold = 7
# input_file = '../Data/ub_sample_data.csv'
# betweenness_output_file = '../Data/mymethod_21.csv'
# community_output_file = '../Data/mymethod_2apr.csv'







start_time = time.time()

# input_file = '../Data/ub_sample_data.csv'
input_data_tmp = sc.textFile(input_file)

# >>> input_data_raw.take(2)
# [['user_id', 'business_id'], ['39FT2Ui8KUXwmUt6hnwy-g', 'RJSFI7mxGnkIIKiJCufLkg']]





header = input_data_tmp.take(1)[0]
input_data_raw = input_data_tmp.filter( lambda x : x!=header).map(lambda x: x.split(",")).map(lambda x: (x[0],x[1]))#.repartition(50)
# >>> input_data_raw.take(1)
# [('39FT2Ui8KUXwmUt6hnwy-g', 'RJSFI7mxGnkIIKiJCufLkg')]


##===================================== Valid Edge Detection ======================================================


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

#input_data =user_business_set.cartesian(user_business_set).\
 #                                       filter(lambda x : x[0][0]!=x[1][0]).\
  #                                      filter(extact_valid_pairs).\
   #                                     map(lambda x: (x[0][0],x[1][0])).distinct().persist()

input_data =user_business_set.cartesian(user_business_set).\
                                        filter(lambda x : x[0][0]!=x[1][0]).\
                                        filter(extact_valid_pairs).\
                                        map(lambda x: tuple(sorted([x[0][0],x[1][0]]))).distinct().persist()


def betweeness_caluclation(node_root, original_graph,num_edges=498):
    '''
    @InputParams : node_root : Root node of tree
    Original Grah : adjacency matrix
    
    Pseudo algorithm
    create the queue que

    label the vertex v as visited and place que

    while que is not empty

    remove the head vertex u of que

    label and enqueue all unvisited neighbors of the vertex u
    
    @return type : Tuple of Node1,Node 2,betweenesss
    '''
    
    queue = []  ## define empty list, we will use to store neighbours 
    bfs_queue = [] ## this will be used to keep note of verteces which have been viisted
    parent_lookup = {} ## basically as we move from one level to lower level in tree, we will keep storing parent node details of each visited node,
    depth_dict={} ### this tells what is depth of visited node 
    srt_pt_cnt = {}   ### this is used to keep count of levels
    betweenness_credit = {}  ### this will calculate credit for each node in a tree
    parent_lookup[node_root] = None  ### since root doesnt have a parent, we hard code as None
    depth_dict[node_root] = 0   #### depth of root is 0
    srt_pt_cnt[node_root] = 1  ### count for root is 1
    bfs_queue.append(node_root)  ## sice we visited root, put it in bfs queue
    try:
        _ = Is_visited_check(node_root,original_graph)  ## just double checking if a node has been visited
    except:
        _=None
    for c in original_graph[node_root]: ### now we first find first level neighbors / child of root node
        parent_lookup[c] = [node_root]  ## assign root node as their parent
        depth_dict[c] = 1 ## define deoth as 1
        srt_pt_cnt[c] = 1
        queue.append(c)  ## out visted node in queue as standard practice in BFS
        bfs_queue.append(c)
    while queue:  ### iterate again and again untile queue is empty
        node = queue.pop(0)  ## pick first child
        betweenness_credit[node] = 1 ## betweenesss of node becomes 1
        try:
            _ = Is_visited_check(node_root,original_graph)
        except:
            _=None
        paths = 0
        if node not in parent_lookup.keys():
            parent_lookup[node]=[]
        for p in parent_lookup[node]:
            if p not in srt_pt_cnt.keys():
                srt_pt_cnt[p] = 0.0
            paths = paths + srt_pt_cnt[p]
        srt_pt_cnt[node] = paths
        for neighbour_nodes in original_graph[node]:
            if neighbour_nodes not in bfs_queue:
                parent_lookup[neighbour_nodes]= [node]
                if node not in depth_dict.keys():
                    depth_dict[node]=0
                if neighbour_nodes not in depth_dict.keys():
                    depth_dict[neighbour_nodes]=0
                depth_dict[neighbour_nodes] = depth_dict[node] + 1
                queue.append(neighbour_nodes)
                bfs_queue.append(neighbour_nodes)
            else:
                if node not in depth_dict.keys():
                    depth_dict[node]= 0
                if depth_dict[neighbour_nodes] == depth_dict[node] + 1:
                    if neighbour_nodes not in parent_lookup.keys():
                        parent_lookup[neighbour_nodes]=[]
                    parent_lookup[neighbour_nodes].append(node)
    bfs_queue.reverse()
    for child_key in bfs_queue[:-1]:
        for parent_key in parent_lookup[child_key]:
            if child_key not in betweenness_credit.keys():
                betweenness_credit[child_key]=0.0
            if parent_key not in srt_pt_cnt.keys():
                srt_pt_cnt[parent_key]=0.0
            if child_key not in srt_pt_cnt.keys():
                srt_pt_cnt[child_key]=0.0
            ratio_prt_cnd = srt_pt_cnt[parent_key] / srt_pt_cnt[child_key]
            score = betweenness_credit[child_key] * ratio_prt_cnd
            if parent_key not in betweenness_credit.keys():
                betweenness_credit[parent_key]=0
            betweenness_credit[parent_key] = score + betweenness_credit[parent_key]
            return_pair = sorted([parent_key,child_key])
            return_pair_tuple = tuple(return_pair)
            yield (return_pair_tuple, score)


def Is_visited_check(node,original_graph):
    neighbours = original_graph.get(node,{})
    num_neighours = 0
    if len(neighbours)>0:
        num_neighours = len(neighbours)
    else:
        num_neighours = 0
    return num_neighours



    
print("Graph generated")
m = input_data.count()  # total edges

print("Total Number of Edges in graph {}".format(m))

### first lets define our adjacency matrix 

'''
1) First reshape input data which typically contains all possible edges in graph into adjacency matrix

Key - Source
Val - List of nodes which can be reached from source as a direct edge

2) store that as a dictionary. We have collectAsMap() function to do so 


'''

bidirectional_edges_rowwise = input_data.flatMap(lambda x: [(x[0], [x[1]]), (x[1], [x[0]])])

original_graph = bidirectional_edges_rowwise.reduceByKey(lambda x, y: x + y).collectAsMap()

vertex_values = []

for i in original_graph.keys():
    vertex_values.append(i)

vertex = sorted(vertex_values)

# >>> len(vertex)
# 222 --- it basically contains list of unique user ids

# original_graph is a dictionary object . each key is a user id, and val is a list of user ids in its adjancency matrix

## x is a node selected everytime and condidered as root of bfs

bgf_poutput  = sc.parallelize(vertex).flatMap(lambda x: betweeness_caluclation(x, original_graph))
betweenness_tmp = bgf_poutput.reduceByKey(add)\
    .map(lambda x: (x[0], x[1])).collect()
betweenness_output = sorted(betweenness_tmp,key=lambda x: (-x[1],x[0][0],x[0][1]))
betweenness = sorted(betweenness_tmp,key=lambda x: (-x[1],x[0][0],x[0][1]))

# >>> betweenness_output[:5]
# [(('cyuDrrG5eEK-TZI867MUPA', 'l-1cva9rA8_ugLrtSdKAqA'), 8468.0)


with open(betweenness_output_file, 'w') as fbetween:
    cnt = 0
    for iter in range(len(betweenness_output)):
        i = betweenness_output[iter]
        if iter == len(betweenness_output)-1:
            fbetween.write("(\'" + str(i[0][0] + "\', \'" + str(i[0][1]) + "\'), " + str(round(i[1]/2,5))))
        else:
            fbetween.write("(\'" + str(i[0][0] + "\', \'" + str(i[0][1]) + "\'), " + str(round(i[1]/2,5)) + "\n"))

fbetween.close()

print("Betweeness File written successfully")




# >>> input_data.count()
# 498    


##============================================================================================================


##===================================== Betweenness Calculation ===============================================



##===================================== Betweenness Calculation Completed =====================================



##===================================== Community Detection Begins ============================================

# original_graph['cyuDrrG5eEK-TZI867MUPA']


# (('cyuDrrG5eEK-TZI867MUPA', 'l-1cva9rA8_ugLrtSdKAqA'), 8468.0)
def geenrate_community(node, neighbouring_nodes_matrix,num_egdes = 498):
    '''
    @Input Params : Root Node, Adjaceny matrix, number of edges

    Logic:
    First find level one nodes. put those node names in a set named used nodes
    Itertatively look for level 1 neighbors of child nodes and keep adding to used nodes. 
    When all neighbours are visited, break the loop

    Return: Set containing nodes in community
    '''
    count = 0
    node_used_set = set()
    neighbour_nodes = neighbouring_nodes_matrix[node]
    community = set()
    while (1):
        node_used_set = node_used_set | neighbour_nodes
        count = 1+count
        new_nodes = set()
        for n in neighbour_nodes:
            new_adj_nodes = neighbouring_nodes_matrix[n]
            new_nodes = new_nodes | new_adj_nodes
        new_used_nodes = node_used_set | new_nodes
        commuhity_length = len(node_used_set)
        current_epoch_neighbours = len(new_used_nodes)
        if  commuhity_length== current_epoch_neighbours:
            break
        neighbour_nodes = new_nodes - node_used_set
    community = node_used_set
    if community == set():
        return {node}
    return community

def community_size(node_list):
    if len(node_list)>10:
        return 10
    else:
        return 1

def com_gen_loop(node, vertexx_list, neighbouring_nodes_matrix,num_edges = 498):
    '''
    @Input Params: rootnode, set of vertices, adjacency matrix, number of edges

    Logic : 
    Initialize emtpy community list.

    Generate 1 community. members of that community become part of unused nodes set

    iterate until all nodes are visited

    '''
    communities_list = []
    
    used_nodes_list = geenrate_community(node, neighbouring_nodes_matrix)
    _ = community_size(used_nodes_list)
    remaining_nodes = vertexx_list - used_nodes_list
    communities_list.append(used_nodes_list)
    while (1):
        root_node_gen = random.sample(remaining_nodes, 1)[0]
        new_used_nodes = geenrate_community(root_node_gen, neighbouring_nodes_matrix)
        _ = community_size(used_nodes_list)
        communities_list.append(new_used_nodes)
        used_nodes_list = used_nodes_list | new_used_nodes
        remaining_nodes = vertexx_list - used_nodes_list
        if len(remaining_nodes) == 0:
            break
    return communities_list


def Q_factor_calc(society, m,default_Q = -1):
    denominator = 2 * m
    if default_Q==-1:
        _=None
    cur_modularity = 0
    for family in society:
        tree_modula = 0
        for i_node in family:
            for j_node in family:
                edge_exists = A[(i_node, j_node)]
                di = degree[i_node]
                dj = degree[j_node]
                overall_degress = dj*di
                tree_modula = tree_modula+ edge_exists - (overall_degress/(denominator))
        cur_modularity = cur_modularity+tree_modula
    return cur_modularity / denominator


# >>> input_data_raw.take(2)
# [['user_id', 'business_id'], ['39FT2Ui8KUXwmUt6hnwy-g', 'RJSFI7mxGnkIIKiJCufLkg']]


edges_tmp = input_data.collect()
edges  = set()

for pair in edges_tmp:
    edges.add(tuple((pair[0],pair[1])))
    edges.add(tuple((pair[1],pair[0])))
    
len(edges)



vertices = set(vertex)

degree = {}

neighbouring_nodes_matrix = defaultdict(set)


for node_pair_tuple in edges:
    neighbouring_nodes_matrix[node_pair_tuple[0]].add(node_pair_tuple[1])
    

# >>> degree
# {'DgfsJqg_gozVgaeZ5vjllA': 14,

def pair_freq(pair):
    if len(pair)>2:
        return 2
    else:
        return 1

# adjacent matrix mapping
A = {}
for n1 in vertex:
    for n2 in vertex:
        pair_nodes = (n1, n2)
        try:
            _ = pair_freq(pair_nodes)
        except:
            _=None
        if pair_nodes in edges:
            A[pair_nodes] = 1
        else:
            A[pair_nodes] = 0

# edge number of the original graph m
m_tmp = len(edges) 


def update_adjacency(key_1,val_1):
    neighbouring_nodes_matrix[key_1].remove(val_1)

'''

Psuedo code for identifying communities

1) FIrst calculate betweeness of each edge in whole graph
2) Now, as per girvan newmen algorithm, remove edges with highlest betweeness
3) for remaining graph, start builgin communities
4) once an node is used in a community, it cant be used in another community in same iteration
5) once all nodes are visited, calculate Q modularity of all communities
6) using logic of finidng max in a search space, identify best iteration where q was highlest and use list of communities in that iteraton

'''

m= m_tmp/ 2

left_edges = m
optimal_modularity = -1


for k, v in neighbouring_nodes_matrix.items():
    degree[k] = len(v)



def community_quality(community_current_iter):
    vect = set()
    number_com = 0
    community_nodes = list()
    for i in community_current_iter:
        vect.add(tuple(i))
        community_nodes.append(i)
    if len(community_nodes)>10:
        number_com = len(community_nodes)
    else:
        number_com = len(community_nodes)
    return number_com




while (1):
    valid_pairs = [w for w in betweenness if w[1]==betweenness[0][1]]
    for pair in valid_pairs:
        key_1 = pair[0][0]
        key_2 = pair[0][1]
        val_1 = pair[0][1]
        val_2 = pair[0][0]
        update_adjacency(key_1,val_1)
        update_adjacency(key_2,val_2)
        left_edges -= 1
    starting_node = random.sample(vertices, 1)[0]
    community_current_iter = com_gen_loop(starting_node, vertices, neighbouring_nodes_matrix)
    try:
        community__qual = community_quality(community_current_iter)
        #print("TRY")
    except:
        community__qual=-1
    modularity_current_iteration = Q_factor_calc(community_current_iter, m)
    if modularity_current_iteration > optimal_modularity:
        optimal_modularity = modularity_current_iteration
        print("Found New Best Community with Q : {} and Number of Communities {}".format(modularity_current_iteration,len(community_current_iter)))
        communities = community_current_iter
    if left_edges == 0:
        break
    bgf_poutput  = sc.parallelize(vertex).flatMap(lambda x: betweeness_caluclation(x, neighbouring_nodes_matrix))
    betweenness_tmp = bgf_poutput.reduceByKey(lambda x, y: x+y)\
        .map(lambda x: (x[0], x[1])).collect()
    betweenness_output = sorted(betweenness_tmp,key=lambda x: (-x[1],x[0][0],x[0][1]))
    betweenness = sorted(betweenness_tmp,key=lambda x: (-x[1],x[0][0],x[0][1]))

sorted_communities = sc.parallelize(communities) \
.map(lambda x: sorted(x)) \
.sortBy(lambda x: (len(x), x)).collect()

with open(community_output_file, 'w+') as fout:
    for community in sorted_communities:
        out_strig = ""
        for nodes in community:
            out_strig+="'"+nodes+"', "
        fout.write(out_strig[:-2] + '\n')

end = time.time()
print('Duration: {}'.format(end - start_time))

