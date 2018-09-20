import numpy as np
import matplotlib.pyplot as plt

def getdate(path):
    """
从本地磁盘获取数据，并输出为样本：X，Y， cluster
    :param path:磁盘路径
    """
    dataset = np.loadtxt(path,encoding='utf-8-sig', delimiter=',')
    X = dataset[:, 1:3]
    m, n = np.shape(X)
    # X0 = np.ones((m, 1))
    # X = np.hstack((X0, X1))
    Y = dataset[:, -1]
    rs = []
    for i in range(len(Y)):
        if Y[i] not in rs:
            rs.append(Y[i])
    cluster = np.array(rs)
    return X, Y, cluster


def class_mean(X, Y, cluster):
    """
定义类别均值
    :param x:样本向量
    :param y: 标签向量
    :param cluster:类别数/簇数/集群数
    """
    mean_vector = []
    for cl in cluster:
        mean_vector.append(np.mean(X[Y == cl], axis=0))  # column mean
    return mean_vector


def within_class_Sw(X, Y, cluster_num):
    """
计算类内散度Sw
    :param x:
    :param y:
    :param cluster_num:
    """
    k = X.shape[1] # 数据属性的数量
    Sw = np.zeros((k,k))
    mean_vectors = class_mean(X, Y, cluster)
    # 协方差矩阵的实现方式，用for循环实现每个类的协方差矩阵再矩阵加法累加
    for cl, mv in zip(cluster, mean_vectors): # cl代表每一类，mv代表每一类的均值向量
        class_sc_mate = np.zeros((k, k)) # 生成临时协方差矩阵
    # 对每个类中的每个（样本向量-对应均值向量）进行矩阵乘法求，累加以求出属于此类的协方差矩阵
        for row in X[Y == cl]:
            sub_tem = (row - mv).reshape(2,1)
            class_sc_mate += np.dot(sub_tem, sub_tem.T)
        Sw += class_sc_mate # 总协方差矩阵
    return Sw


'''
        x_tmp = X[i].reshape(k, 1)  # row -> cloumn vector
        if Y[i] == 1: u_tmp = mena_vecoors[1].reshape(n, 1)
        if Y[i] == 0: u_tmp = mean_vectors[0].reshape(n, 1)
        return Sw += np.dot( x_tmp - u_tmp, (x_tmp - u_tmp).T )
'''


def between_class_Sb(X, Y, cluster):
    """
计算类内散度
    :param X:
    :param Y:
    :param cluster:
    """
    k = X.shape[1]
    Sb = np.zeros((k,k))
    mean_vector_all = np.mean(X, axis=0)
    mean_vectors = class_mean(X, Y, cluster)
    # for mean_vec in enumerate(mean_vectors):
    for cl, mean_vec in zip(cluster,mean_vectors):
        mi = X[Y == cl].shape[0] # 每个类的样本数量
        sub_tem = (mean_vec - mean_vector_all).reshape(k, 1)
        Sb += mi * np.dot(sub_tem, sub_tem.T)
    return Sb


def get_w_2c(Sw):
    """
对于二分类问题的求法
    :param Sw:
    :return:
    """
    k = Sw.shape[0]
    Sw = np.mat(Sw)
    U, sigma, V = np.linalg.svd(Sw) #奇异值分解求Sw的逆
    # 如果矩阵不可逆则：
    # 方法一：是在矩阵对角线追加极小值，Sw = Sw + rI，I是单位矩阵，r是一个极小的数，使其可逆
    # 方法二：先对数据使用PCA对数据进行降维使其可逆。
    # 这里还没学到PCA，就使用方一
    try:
        Sw_inv = V.T * np.linalg.inv(np.diag(sigma)) * U.T
    except LinAlgError as reason:
        print('不可逆： ', str(reason))
        I = np.eye(k)
        Sw += 1e-9 * I
        Sw_inv = V.T * np.linalg.inv(np.diag(sigma)) * U.T
    w = np.dot(Sw_inv, (mean_vectors[0] - mean_vectors[1]).reshape(k, 1))
    w = np.asarray(w).reshape(k,)
    return w

def get_W_mc(Sw, Sb, cluster):
    """
对于多类问题
    :param Sw:
    :param Sb:
    """
    k = Sw.shape[0]
    c = len(cluster)
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb)) #特征向量是k行k维，一行是一个特征向量
    # for w, v in zip(eig_vals, eig_vecs):
    #     eigvec_sc = eig_vecs[:, i].reshape(k,1) #每个特征值对应的特征向量
    eig_pairs = zip(eig_vals, eig_vecs)
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    W = eig_pairs[0][1].reshape(k, 1)
    for i in range(1, c-1):
        Wi = eig_pairs[i][1].reshape(k, 1)
        W = np.hstack((W, Wi))
    return W




X, Y, cluster = getdate('D:\\Desktop\西瓜数据.csv')
Sw = within_class_Sw(X, Y, cluster)
Sb = between_class_Sb(X, Y, cluster)
mean_vectors = class_mean(X, Y, cluster)
w = get_w_2c(Sw)
W = get_W_mc(Sw, Sb, cluster)
print(Sw)
print(Sb)
print(w)
print(W)

fig = plt.figure()
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.title('watermelon_3a')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], marker='o', color='r', label='good')
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], marker='o', color='g', label='bad')
x = np.linspace(0.2, 0.8)
plt.plot(x, w[0] * x/w[1], 'y-', lw=1)
plt.legend()
plt.show()


