import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import math
import numpy as np
import operator
from sklearn import preprocessing
import uuid
import pickle
from collections import defaultdict, namedtuple
from sklearn.tree import export_graphviz
import pydotplus

df=pd.read_excel('\decision tree-data.xlsx',sheetname='original data')


def infogain(df,attribute,target):#df is the whole dataframe, attribute is one of attribute name, target is target name
    smalldf=df[[attribute,target]]
    grouped= pd.crosstab(smalldf[attribute], smalldf[target]).stack()#statistic by group keeping 0
    attribute_list=list(set(df[attribute]))
    target_list=list(set(df[target]))
    mid_entropy=[]
    count_sum = []  # record each attribute count
    for i in range(len(attribute_list)):
        sum=0
        for j in range(len(target_list)):
                sum=grouped[attribute_list[i]][target_list[j]]+sum #count total
        info_entropy=0
        for j in range(len(target_list)):
            if grouped[attribute_list[i]][target_list[j]]!=0:
                info_entropy=info_entropy+(-1)*grouped[attribute_list[i]][target_list[j]]/sum*math.log(grouped[attribute_list[i]][target_list[j]]/sum,2)
            else:
                info_entropy=info_entropy+0
        count_sum.append(sum)
        if pd.isnull(info_entropy):#prevent 0 log
            mid_entropy.append(0)
        else:
            mid_entropy.append(info_entropy)
    Total_entropy=0
    s=0#record total count
    for i in range(len(count_sum)):
        s=s+count_sum[i]
    for i in range(len(count_sum)):
        Total_entropy=Total_entropy+count_sum[i]/s*mid_entropy[i]
    tgrouped = smalldf.groupby(by=[target],as_index=False)
    tentropy=0#original entropy
    for i in range(len(target_list)):
        tentropy=tentropy+(-1)*len(tgrouped.get_group(target_list[i]))/s*math.log(len(tgrouped.get_group(target_list[i]))/s,2)
    return tentropy-Total_entropy

def rank_attribute(df,target):#scan all attribute
    col=df.columns
    entropy_list=[]
    for i in range(len(col)):
        if col[i]!=target:
           entropy_gain= infogain(df, col[i], target)
           entropy_list.append(entropy_gain)
    max_attribute=col[entropy_list.index(max(entropy_list))]#find attribute with highest entropy
    # print(col,entropy_list)
    return max_attribute


#split the dataset by value
def splitdataset(dataset, axis, value):
    retdataset = dataset[dataset[axis] == value]
    del retdataset[axis]
    return retdataset

#determine the split attribute by majority
def majority(labellist):
    classcount = {}
    #vote for each label
    for vote in labellist:
        if vote not in classcount.keys():
            classcount[vote] =0
        classcount[vote] += 1
    #rank the class
    sortedclasscount = sorted(classcount.iteritems(), key=operator.itemgetter(1), reverse=True)
    #return majority
    return sortedclasscount[0][0]

def createtree(dataset, result):
    '有2个参数，第一个是dataFrame类型的数据，第个是字符串类型的y标签'
    #如果数据集的分类只有一个，则返回此分类，无需子树生长
    classlist = list(dataset[result].values)
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]

    #如果数据集仅有1列变量，加y标签有2列，则返回此数据集的分类
    if len(dataset.columns) == 2:
        return majority(classlist)

    bestfeat = rank_attribute(dataset,result)
    mytree = {bestfeat: {}}
    #此节点分裂为哪些分枝
    uniquevals = set(dataset[bestfeat])

    #对每一分枝，递归创建子树
    for value in uniquevals:
        mytree[bestfeat][value] = createtree(splitdataset(dataset, bestfeat,value),result)
    #完成后，返回决策树
    return mytree


def get_nodes_edges(tree=None, root_node=None):
    ''' 返回树中所有节点和边
    '''
    Node = namedtuple('Node', ['id', 'label'])
    Edge = namedtuple('Edge', ['start', 'end', 'label'])

    if type(tree) is not dict:
        return [], []
    nodes, edges = [], []
    if root_node is None:
        label = list(tree.keys())[0]
        root_node = Node._make([uuid.uuid4(), label])
        nodes.append(root_node)
    for edge_label, sub_tree in tree[root_node.label].items():
        node_label = list(sub_tree.keys())[0] if type(sub_tree) is dict else sub_tree
        sub_node = Node._make([uuid.uuid4(), node_label])
        nodes.append(sub_node)
        edge = Edge._make([root_node, sub_node, edge_label])
        edges.append(edge)
        sub_nodes, sub_edges = get_nodes_edges(sub_tree, root_node=sub_node)
        nodes.extend(sub_nodes)
        edges.extend(sub_edges)
    return nodes, edges

def dotify(tree):
    ''' 获取树的Graphviz Dot文件的内容
    '''

    content = 'digraph decision_tree {\n'
    nodes, edges = get_nodes_edges(tree)
    for node in nodes:
        content += '    "{}" [label="{}"];\n'.format(node.id, node.label)
    for edge in edges:
        start, label, end = edge.start, edge.label, edge.end
        content += '    "{}" -> "{}" [label="{}"];\n'.format(start.id, end.id, label)
    content += '}'
    return content

def load_tree(filename):
    ''' 加载树结构
    '''
    with open(filename, 'rb') as f:
        tree = pickle.load(f)
    return tree


def classify(data_vect, feat_names=None, tree=None):
    ''' 根据构建的决策树对数据进行分类
    '''
    if tree is None:
        tree = self.tree
    if feat_names is None:
        feat_names = self.feat_names
    # Recursive base case.
    if type(tree) is not dict:
        return tree
    feature = list(tree.keys())[0]
    value = data_vect[feat_names.index(feature)]
    sub_tree = tree[feature][value]
    return classify(feat_names, data_vect, sub_tree)


if __name__ == '__main__' :
    tree_list=[]#save tree
    mytree=createtree(df,'Conflict')
    print(mytree)
    with open('\traffic conflict.dot', 'w') as f:
        dot = dotify(mytree)
        f.write(dot)

    graph = pydotplus.graph_from_dot_data(dot)#可视化树
    # 保存图像到pdf文件
    graph.write_pdf("\newtraffic conflict.pdf")
