import numpy as np

class PCA():
    def __init__(self, d):
        """
        传入要保留的维数
        :param d:
        """
        self.d = d

    def feature_axtraction(self, X):
        """
        计算PCA后的矩阵X
        :param X: (n, m)
        由于X不是方阵，改由奇异值分解求取其特征值和特征向量
        """
        # 求出XX.T的特征值及特征向量
        '''用特征向量做'''
        # P = np.dot(X, X.T)
        # eig_values, eig_vector = np.linalg.eig(P)  # 计算P的特征值和特征向量
        # idx = np.argsort(-1 * eig_values)
        # self.eig_val = eig_values[idx][: self.d]
        # self.eig_vec = eig_vector[:, idx][:, :self.d]  # 列向量是特征向量
        # lowDData_X = np.dot(self.eig_vec.T, X)
        '''用奇异值分解做'''
        print(X.shape)
        U, s, V = np.linalg.svd(X)
        eig_values = s ** 2
        idx = np.argsort(-1 * eig_values)
        self.eig_val = eig_values[idx][: self.d]
        self.eig_vec = U[:, idx][:, :self.d] # 行向量是特征向量
        # 将 X映射到低维空间
        lowDData_X = np.dot(self.eig_vec.T, X)

        return lowDData_X

def normalize(X):
    """
    中心化
    :param X: (n,m)维
    """
    mean_vec = np.mean(X, axis=0)
    normalized_X = X - mean_vec
    return normalized_X

''' 导入数据'''
# 构造文件打开路径
path = 'F:/IT学习/机器学习/西瓜书/算法实践/Unit10/yalefaces'
import os
from PIL import Image, ImageDraw
X = []
for file in os.listdir(path):
    if not file.endswith('.txt'):
        img = Image.open(os.path.join(path, file))
        img = np.array(img).reshape(img.width * img.height)  # 图片为灰度图片，模式为L
        X.append(np.array(img))
X = normalize(np.array(X)).T

'''主成份分析'''
d = 20
pca = PCA(d)
new_X = pca.feature_axtraction(X)

new_img = Image.fromarray(new_X).convert('L') # 将降维后的数据保存为Image对象
new_file = str(d) + '_' + dirs[0].split('.')[0] + '.png'
new_img.save(os.path.join('yalefaces', new_file))
