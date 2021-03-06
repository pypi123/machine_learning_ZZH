import pandas as pd

def calShannonEnt(df):
    """
计算当前节点的信息熵
    :param df: DataFrame 数据集
    """
    import numpy as np
    groub_by_label = df.groupby(df.columns[-1])
    Entries = 0.0
    for label, group in groub_by_label:
        prob = len(group)/len(df)
        Entries += -prob * np.log2(prob)

    return Entries


def chose_feature(df):
    features = df.columns[:-1]
    baseEnt = calShannonEnt(df)
    Gain = 0.0
    feature_chosed = features[0]
    T_num = len(df) - 1
    T_dic = {}
    for feature in features:
        if df[feature].dtype == 'object':
            group_by_feature = df.groupby(feature)
            Ent_sum = 0.0
            for value, group in group_by_feature:
                Ent = calShannonEnt(group)
                Ent_sum += (len(group)/len(df))*Ent
            Gain_feature = baseEnt - Ent_sum
            # print('当前属性：', feature)
            # print('当前属性的增益：', Gain_feature)
        else:
            # 对于连续型属性，先将df进行排序
            df = df.sort_values(by=feature, axis=0, ascending=True).reset_index(drop=True)
            Gain_feature = 0.0
            T_dic[feature] = 0.0  # 用字典型来存储多个连续属性的最优分割值，避免分割值永远是最后一列
                                  # 连续属性的分割值
            # T = 0.0  # T是分割值
            feature_values = []  # 对于连续型的属性，将属性值记录到列表里
            for value in df[feature]:
                feature_values.append(value)
            for i in range(T_num):
                T_temp = (feature_values[i] +feature_values[i+1]) / 2
                group_small = df[df[feature] <= T_temp]
                group_big = df[df[feature] > T_temp]
                Ent_small = calShannonEnt(group_small)
                Ent_big = calShannonEnt(group_big)
                p_small = len(group_small)/len(df)
                p_big = len(group_big)/len(df)
                Ent_sum = p_small*Ent_small + p_big*Ent_big
                Gain_temp = baseEnt - Ent_sum
                # print('第{}次分割'.format(i+1))
                # print('当前分割值：', T_temp)
                if Gain_temp > Gain_feature:
                    # print('T{}->{}'.format(T_dic[feature], T_temp))
                    # print('Gain{}->{}'.format(Gain,Gain_temp))
                    Gain_feature = Gain_temp
                    T_dic[feature] = T_temp  # 对于两列的连续值。T永远是最后一列连续值的值，所以出错,
                                             # 改进用字典来存储

        if Gain_feature > Gain:
            Gain = Gain_feature
            feature_chosed = feature

    if df[feature_chosed].dtype == object:
        T = None
    else:
        T = T_dic[feature_chosed]
    return feature_chosed, T

class Node(object):
    def __init__(self,attr_init=None, label_init=None, attr_down_init={}):
        """
    节点类有三个属性：
        :param attr_init: 作为一个新的分支的父类（任何节点都有，页节点为None），
                           当前节点的划分属性，按这个属性划分产生子节点
        :param attr_down_init: 字典,
        关于key，若属性是离散属性，则是具体属性的值
                 若属性是连续属性，则是‘<=分割值’或'>分割值'
                 分割值：对于样本中的的连续属性a，其按小到大排列后，
                 每个相邻的值相加后除以2作为一个候选的分割值，一共有n-1个，
                 分别考察这n-1个分割值分割后产生的二分类样本的信息增益，
                 取其中能使信息增益最大的候选分割点作为分割点。
                 信息增益Gain(D, a) = Gain(D, a, t).max [t是分割点]
                                    = Ent(D) - (D[+/-]t/D) * Ent(Dt).sum(t0, t(n-1))
                                    D[+/-]D代表按t分割后的样本D-(value<=分割值)和D+(value>分割值)
                         Ent(D) = -pk * log2(pk) -----pk是当前样本集合中属于第k类样本所占的比例
        value：是子节点
        :param label_init: 节点的类标签数据（此节点中属于最多的类的标签）
        """
        self.attr = attr_init
        self.attr_down = attr_down_init
        self.label = label_init
'''
def get_node_label(df):
    groups = df.groupby(df[df.columns[-1]])
    label_num = 0
    label_val = None
    for label, group in groups:
        if len[group] > label_num:
            label_num = len(group)
            label_val = label
    return label_val
'''

def TreeGenerate(df):
    """
决策树生成的框架
    :param df:DataFrame数据
    """
    # print(df)
    new_node = Node(None, None, {})
    label_counts = {}
    for label in df[df.columns[-1]]:
        if label not in label_counts:
            label_counts[label] = 1
        else:
            label_counts[label] += 1
    # print('当前处理数据的标签统计： ', label_counts)
    if len(label_counts) == 1:   # or len(df) == 0 or len(df.columns)-1 == 0:
        new_node.label = df.loc[0,df.columns[-1]]
        new_node.attr = None
        new_node.attr_down = None
        # print('当前节点的标签是：', new_node.label)
        return new_node
    else:
        new_node.label = max(label_counts, key=label_counts.get)
        # print('当前节点的标签是：', new_node.label)
    new_node.attr, div_value = chose_feature(df)
    # print('当前节点的属性是： ', new_node.attr)
    if div_value == None:
        group_by_attr = df.groupby(new_node.attr)
        for value, group in group_by_attr:
            new_node.attr_down[value] = TreeGenerate(group.drop(columns=new_node.attr).reset_index(drop=True))
    else:
        # print('当前节点的划分值是： ', div_value)
        # new_node.attr_down['<={}?'.format(div_value)] = TreeGenerate(df[df[new_node.attr] <= div_value].reset_index(drop=True))
        # new_node.attr_down['>{}?'.format(div_value)] = TreeGenerate(df[df[new_node.attr] > div_value].reset_index(drop=True))
        new_node.attr_down['<=%.3f'% div_value] = TreeGenerate(
            df[df[new_node.attr] <= div_value].reset_index(drop=True))
        new_node.attr_down['>%.3f'% div_value] = TreeGenerate(
            df[df[new_node.attr] > div_value].reset_index(drop=True))

    return new_node

'''决策树可视化实现'''
def TreeToGraph(i, father_node, g):
    """
    给定起始节点的名字i（用数字记录节点名）、节点、和 标签名字
    用i+1，和子节点及标签名作为递归输入
    返回的是i和子节点的名称
    将所有的节点遍历完成后返回
    :param i: 为了避免迭代时子节点重新从零开始计，这里传入参数i用来累加迭代
    :param node:根节点
    :param df:根节点的数据
    """
    from pydotplus import graphviz
    if father_node.attr == None:
        node_label = 'Node: %d\n好瓜: %s'% (i, father_node.label)
    else:
        node_label = 'Node: %d\n好瓜: %s\n属性: %s' % (i, father_node.label, father_node.attr)
    father_node_name = i
    node_graph_obj = graphviz.Node(father_node_name, label=node_label, fontname='SimHei')  # 创建graphviz.Node对象
    g.add_node(node_graph_obj)  # 将创建的graphviz节点对象添加到graphviz点图Dot对象
    if father_node.attr != None:
        for value in father_node.attr_down:
            child_node = father_node.attr_down[value]
            i, child_node_name = TreeToGraph(i+1, child_node, g)
            g_edge = graphviz.Edge(father_node_name, child_node_name, label=value, fontname='SimHei')

            # 创建edge对象，将父节点与子节点进行连接
            g.add_edge(g_edge)  # 将edge对象加入到点图dot对象中
    return i, father_node_name

# 用graphviz实现
with open('D:\\Desktop\西瓜数据集3.0.csv') as data_file:
    df = pd.read_csv(data_file)
Tree = TreeGenerate(df)
from pydotplus import graphviz
g = graphviz.Dot() # 创建一个Dot图对象
TreeToGraph(0, Tree, g)
g2 = graphviz.graph_from_dot_data(g.to_string()) # 将Dot对象输出为字符串g.to_string()
                                                 # 并通过graphviz解码
g2.write_png('D:\\Desktop\ID3test.png')


'''
这里在摸索框架
def graph(father_node,i):
    father_node_name = i
    if father_node.attr == None:
        print('页节点: %d\n类别: %s' % (i, father_node.label))
        return i
    else:
        print('节点: %d\n类别: %s' % (i, father_node.label))
        for value in father_node.attr_down:
            child_node = father_node.attr_down[value]
            i = graph(child_node, i+1)

# for value in Tree.attr_down:
#     node2 = Tree.attr_down[value]
#     if node2.attr == None:
#         print('父节点：%s -> 页节点：%s' % (Tree.attr, value))
#     else:
#         print('父节点：%s -> 子节点：%s' % (Tree.attr, value))
#         print('最佳分割属性：', node2.attr)
#         for value in node2.attr_down:
#             node3 = node2.attr_down[value]
#             if node3.attr == None:
#                 print('父节点：%s -> 页节点：%s' % (node2.attr, value))
#             else:
#                 print('父节点：%s -> 子节点：%s' % (node2.attr, value))
#                 print('最佳分割属性：', node3.attr)
'''


'''用Digraph包实现

def TreeToGraph(i, father_node, dot):
    """
    给定起始节点的名字i（用数字记录节点名）、节点、和 标签名字
    用i+1，和子节点及标签名作为递归输入
    返回的是i和子节点的名称
    将所有的节点遍历完成后返回
    :param i: 为了避免迭代时子节点重新从零开始计，这里传入参数i用来累加迭代
    :param node:根节点
    :param df:根节点的数据
    """
    from graphviz import Digraph
    if father_node.attr == None:
        node_label = 'Node: %d\n好瓜: %s'% (i, father_node.label)
    else:
        node_label = 'Node: %d\n好瓜: %s\n属性: %s' % (i, father_node.label, father_node.attr)
    father_node_name = str(i)  # dot对象添加node时，node的名字需要用字符串或者字节表示，
                               # 想不通为什么
    dot.node(name=father_node_name, label=node_label, fontname='SimHei', shape='rect')  # 创建节点
    if father_node.attr != None:
        for value in father_node.attr_down:
            child_node = father_node.attr_down[value]
            i, child_node_name = TreeToGraph(i+1, child_node, dot)
            dot.edge(tail_name=father_node_name, head_name=child_node_name, label=value, fontname='Simhei')

    return i, father_node_name

with open('D:\\Desktop\西瓜数据集3.0.csv') as data_file:
    df = pd.read_csv(data_file)
feature = df.columns[:-1]
Tree = TreeGenerate(df)

import os
from graphviz import Digraph
dot = Digraph(comment='test')
TreeToGraph(0, Tree, dot)
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'
dot.render('./decession_tree_test.gv', view=True)
'''
def predict(Tree, df_sample):
    """
预测单个样本的标签
    :param Tree:
    :param df_sample: 单个样本
    :return:
    """
    import re
    if Tree.attr == None:
        # print(Tree.label)
        return Tree.label
    value = df_sample[Tree.attr]
    if type(value) == str:
        new_node = Tree.attr_down[value]
        label = predict(new_node, df_sample)
    else:
        for key in Tree.attr_down:
            div_num = float(re.findall(r'\d+\.?\d+', key)[0])
            break
        if value <= div_num:
            key = '<%.3f' % div_num
            new_node = Tree.attr_down[key]
            label = predict(new_node, df_sample)
        else:
            key = '>%.3f' % div_num
            new_node = Tree.attr_down[key]
            label = predict(new_node, df_sample)
    return label

def predict_accuracy(Tree, df_test):

    accuracy_num = 0
    for index in df_test.index:
        test_sample = df_test.loc[index, :]  # 转换为Series对象
        label = predict(Tree, test_sample)
        if label == test_sample[test_sample.index[-1]]:
            accuracy_num += 1
    accuracy = accuracy_num/len(df_test)
    return accuracy


df_test_set = df.loc[1:5, :]
# index = df_sample.index[-1]
# index1 = df.index
# print(df_sample[index])
# print(index1)
# a = predict(Tree, df_sample)
# print(a)
acc = predict_accuracy(Tree, df_test_set)
print(acc)
