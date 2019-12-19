import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import math
import numpy as np
import operator
from sklearn import preprocessing

df=pd.read_excel('decision tree-data.xlsx',sheetname='original data')


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


if __name__ == '__main__' :
    tree_list=[]#save tree
    mytree=createtree(df,'Conflict')
    print(mytree)

