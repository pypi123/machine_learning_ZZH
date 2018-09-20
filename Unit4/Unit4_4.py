import pandas as pd

def Get_Gini(df):
    groub_by_label = df.groupby(df.columns[-1])
    Gini = 1
    for label, group in groub_by_label:
        prob = len(group) / len(df)
        Gini -= prob**2
    return Gini

def chose_feature_Gini(df):
    features = df.columns[:-1]
    feature_chosed = features[0]
    T_num = len(df) - 1
    T_dic = {}
    Gini_index = 0.0
    for feature in features:
        if df[feature].dtype == 'object':
            group_by_feature = df.groupby(feature)
            Gini_index_feature = 0.0
            for value, group in group_by_feature:
                r = len(group)/len(df)
                Gini_index_feature += r * Get_Gini(group)
        else:
            df = df.sort_values(by=feature, axis=0, ascending=True).reset_index(drop=True)
            Gini_index_feature = 0.0
            T_dic[feature] = 0.0
            feature_values = []
            for value in df[feature]:
                feature_values.append(value)
            for i in range(T_num):
                T_temp = (feature_values[i] + feature_values[i + 1]) / 2
                group_small = df[df[feature] <= T_temp]
                group_big = df[df[feature] > T_temp]
                Gini_small = Get_Gini(group_small)
                Gini_big =Get_Gini(group_big)
                p_small = len(group_small) / len(df)
                p_big = len(group_big) / len(df)
                Gini_index_temp = p_small*Gini_small + p_big*Gini_big
                if Gini_index_temp < Gini_index_feature:
                    Gini_index_feature = Gini_index_temp
                    T_dic[feature] = T_temp

        if Gini_index_feature < Gini_index:
            Gini_index = Gini_index_feature
            feature_chosed = feature
    if df[feature_chosed].dtype == object:
        T = None
    else:
        T = T_dic[feature_chosed]
    return feature_chosed, T


class Node(object):
    def __init__(self,attr_init=None, label_init=None, attr_down_init={}):
        self.attr = attr_init
        self.attr_down = attr_down_init
        self.label = label_init


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
    new_node.attr, div_value = chose_feature_Gini(df)
    # print('当前节点的属性是： ', new_node.attr)
    if div_value == None:
        group_by_attr = df.groupby(new_node.attr)
        for value, group in group_by_attr:
            new_node.attr_down[value] = TreeGenerate(group.drop(columns=new_node.attr).reset_index(drop=True))
    else:
        # print('当前节点的划分值是： ', div_value)
        new_node.attr_down['<={}?'.format(div_value)] = TreeGenerate(df[df[new_node.attr] <= div_value].reset_index(drop=True))
        new_node.attr_down['>{}?'.format(div_value)] = TreeGenerate(df[df[new_node.attr] > div_value].reset_index(drop=True))
    return new_node


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
    else:
        # print('当前处理数据：', df_sample)
        # print('最佳分割属性：', Tree.attr)
        value = df_sample[Tree.attr]
        # print('最佳分割属性下的取值： ', value)
        if type(value) == str:
            if value in Tree.attr_down:   #  这里防止因训练数据过少而损失属性值
                                          #  即两个属性的值不是独立的，导致验证时验证机在此属性的
                                          #  取值没有在节点上出现而出现attr_down的索引错误
                new_node = Tree.attr_down[value]
                label = predict(new_node, df_sample)
            else:
                label =  Tree.label
        else:
            for key in Tree.attr_down:
                print(key)
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
    """
预测精度
    :param Tree:
    :param df_test: 测试集
    :return: 返回精度值
    """
    accuracy_num = 0
    for index in df_test.index:
        test_sample = df_test.loc[index, :]  # 转换为Series对象
        label = predict(Tree, test_sample)
        if label == test_sample[test_sample.index[-1]]:
            accuracy_num += 1
    accuracy = accuracy_num/len(df_test)
    return accuracy

def get_label_counts(df):
    """
获取df数据集的分类情况
    :param df:
    :return: 字典key是类别名
              value是出现的次数
    """
    label_counts = {}
    for label in df_train[df_train.columns[-1]]:
        if label not in label_counts:
            label_counts[label] = 1
        else:
            label_counts[label] += 1
        return label_counts

def Preprune(df_train, df_test):
    """
前剪枝
    :param df_train: 训练集
    :param df_test: 测试集
    :return: 剪枝后的节点类
    """
    new_node = Node()
    label_counts = get_label_counts(df_train)
    if len(label_counts) == 1:
        new_node.label = df.loc[0, df.columns[-1]]
        return new_node
    else:
        new_node.label = max(label_counts, key=label_counts.get)
        acc0 = predict_accuracy(new_node, df_test)
        new_node.attr, div_value = chose_feature_Gini(df)
        if div_value == None:  # 对于离散属性
            group_by_attr = df_train.groupby(new_node.attr)
            for value, group in group_by_attr:
                child_node = Node()
                group_set = group.drop(columns=new_node.attr).reset_index(drop=True) # 离散属性需要删除，重置索引为(0,)
                                                                                     # 为后续获取label时提供方便
                child_label_counts = get_label_counts(group_set)
                child_node.label = max(child_label_counts, key=child_label_counts.get)
                new_node.attr_down[value] = child_node
            acc1 = predict_accuracy(new_node, df_test)
            if acc1 > acc0:  # 需要分支
                for value, group in group_by_attr:
                    group_set = group.drop(columns=new_node.attr).reset_index(drop=True)
                    new_node = Preprune(group_set, df_test)
            else:
                new_node.attr_down = None
                new_node.attr = None
        else:  # 对于连续属性
            group_small = df_train[df_train[feature] <= div_value]
            group_big = df_train[df_train[feature] > div_value]
            i = 0
            for group in [group_small, group_big]:
                child_node = Node()
                group_set = group.reset_index(drop=True) # ；连续性属性不用删除
                child_label_counts = get_label_counts(group_set)
                child_node.label = max(child_label_counts, key = child_label_counts.get)
                if i == 0:
                    new_node.attr_down['<=%.3f' % div_value] = child_node
                    i += 1
                else:
                    new_node.attr_down['>%.3f' % div_value] = child_node
            acc1 = predict_accuracy(new_node, df_test)
            if acc1 > acc0: # 需要分支
                for group in [group_small, group_big]:
                    group_set = group.reset_index(drop=True)
                    new_node = Preprune(group_set, df_test)
            else:
                new_node.attr_down = None
                new_node.attr = None
    return new_node


def node_plie(node_dic, i, tree):
    """
记录每一层的节点
    :param node_dic: 记录用的字典
    :param i: 记录层数
    :param tree: 待记录的决策树
    :return: 返回字典其中key是层数，value是当前层数节点的集合

    """
    if tree.attr == None:
        node_dic.setdefault(i, []).append(tree)
        return node_dic
    else:  # 证明有下一层
        node_dic.setdefault(i, []).append(tree)  # 记录当前层的节点
        i += 1             # 层数加一
        for key in tree.attr_down:
            child_node = tree.attr_down[key]
            node_dic = node_plie(node_dic, i, child_node)
    return node_dic


def postpurn(tree, df_test):
    """
后剪枝
    :param tree: 未剪枝的完全树
    :param df_test: 测试集
    """
    if tree.attr == None:
        return tree
    a0 = predict_accuracy(tree, df_test)
    node_dic = node_plie({}, 0, tree)
    for i in range(len(node_dic)-2):
        nodes_list = node_dic[len(node_dic)-i-2]
        for node in nodes_list:
            if node.attr != None:
                tem_attr = node.attr
                node.attr = None
                a1 = predict_accuracy(tree, df_test)
                if a0 > a1:
                    node.attr = tem_attr
                else:
                    a0 = a1
    return tree

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
    
    
'''用graphviz实现'''
with open('D:\\Desktop\西瓜数据集3.0.csv') as data_file:
    df = pd.read_csv(data_file)
    print(df)

train_index = [0, 1, 2, 5, 6, 9, 13, 14, 15, 16]
test_index = [3, 4, 7, 8, 10, 11, 12]
df_train = df.loc[train_index, :].reset_index(drop=True)
df_test = df.loc[test_index, :].reset_index(drop=True)
print(df_test)
# Tree = Preprune(df_train, df_test)
Tree = TreeGenerate(df_train)
Tree = postpurn(Tree, df_test)
node_plies= node_plie({}, 0, Tree)

from pydotplus import graphviz
g = graphviz.Dot() # 创建一个Dot图对象
TreeToGraph(0, Tree, g)
g2 = graphviz.graph_from_dot_data(g.to_string()) # 将Dot对象输出为字符串g.to_string()
                                                 # 并通过graphviz解码
g2.write_png('D:\\Desktop\postpurn_test.png')

# 第二种可视化实现方式（测试成功）
'''
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
    father_node_name = str(i)  # dot对象添加node时，node的名字需要用字符串或者字节表示
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
dot = Digraph(comment='test1')
TreeToGraph(0, Tree, dot)
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'
dot.render('./decession_tree_CART.gv', view=True)
'''