import pandas as pd
with open('D:\\Desktop\西瓜数据集3.0.csv') as data_file:
    df = pd.read_csv(data_file)
# print(df)
feature = df.columns[:-1]
# print(feature)



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
    for feature in features:
        if df[feature].dtype == 'object':
            group_by_feature = df.groupby(feature)
            Ent_sum = 0.0
            for value, group in group_by_feature:
                Ent = calShannonEnt(group)
                Ent_sum += (len(group)/len(df))*Ent
            Gain_feature = baseEnt - Ent_sum
        else:
            df = df.sort_values(by=feature, axis=0, ascending=True).reindex()
            Gain_feature = 0.0
            T = 0.0  # T是分割值
            feature_values = []
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
                if Gain_temp > Gain_feature:
                    Gain_feature = Gain_temp
                    T = T_temp
        if Gain_feature > Gain:
            Gain = Gain_feature
            feature_chosed = feature

    if df[feature_chosed].dtype == object:
        T = None

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
    new_node = Node(None, None, {})
    label_counts = {}
    for label in df[df.columns[-1]]:
        if label not in label_counts:
            label_counts[label] = 1
        else:
            label_counts[label] += 1
    new_node.label = max(label_counts, key=label_counts.get)
    print(new_node.label)
    if len(label_counts) == 1 or len(df) == 0 or len(df.columns)-1 == 0:
    # groups = df.groupby(df[df.columns[-1]])
    # label_num = 0  # 将此节点中属于某类的样本数量初始化为0
    # if len(groups) > 1:
    #     for label, group in groups:
    #         if len(group) > label_num:
    #             label_num = len(group)
    #             new_node.label = label
    # else:
    #     new_node.label = df[df.columns[-1]][0]
        return new_node
    new_node.attr, div_value = chose_feature(df)
    if div_value == None:
        group_by_attr = df.groupby(new_node.attr)
        for value, group in group_by_attr:
            new_node.attr_down[value] = TreeGenerate(group.drop(columns=new_node.attr))
    else:
        new_node.attr_down['<={}?'.format(div_value)] = TreeGenerate(df[df[new_node.attr] <= div_value])
        new_node.attr_down['>{}?'.format(div_value)] = TreeGenerate(df[df[new_node.attr] > div_value])
    return new_node

# Ent = calShannonEnt(df)
# feature = chose_feature(df)
# print(Ent)
# print(feature)
Tree = TreeGenerate(df)
print(Tree.attr_down)
