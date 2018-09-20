import numpy as np

class Net():
    def __init__(self):
        # 初始化层数
        self.i_n = 0
        self.h_n = 0
        self.o_n = 0
        # 初始化各层的值
        self.i_v = np.zeros(self.i_n)
        self.h_v = np.zeros(self.h_n)
        self.o_v = np.zeros(1)
        self.w = []
        self.beta = []
        self.c = []
        # 初始化误差
        self.err = 0.0

    def Create_nn(self, h_n, centers, learninggrate=0.1):
        self.h_n = h_n
        self.h_v = np.zeros(self.h_n)
        self.c = centers
        self.lr = learninggrate
        self.w = np.random.rand(self.h_n)
        self.beta = np.random.rand(self.h_n)

    def predict(self, x):
        # x是(1,i_n)维的
        for i in range(self.h_n):
            beta = self.beta[i]          # beta 是1维的
            c = self.c[i]                   # c是(1,i_n)维的
            self.h_v[i] = RBF(x, beta, c)
        self.o_v = np.dot(self.h_v, self.w.T) # self.h_v 是(1,h_n), self.w (1, h_n)
        return self.o_v   # 1维

    def RBF_trainer(self, inputset, outputset):
        self.i_v = inputset
        self.labelset = outputset
        self.err = 0.0
        err_temp = 0.0
        for i in range(self.i_v.shape[0]):
            sample = self.i_v[i]
            label = self.labelset[i]
            self.o_v = self.predict(sample)
            err_temp += (self.o_v - label)**2
            # 定义梯度项
            self.grid = (self.o_v - label) * self.h_v  # （1，h_n）
            # 更新参数
            for i in range(self.h_n):
                c = self.c[i]
                grid = self.grid[i]
                w = self.w[i]
                self.beta[i] += self.lr * grid * w * np.sqrt(np.dot((sample - c), (sample - c).T))
            self.w -= self.lr * self.grid
        self.err = err_temp / len(self.i_v)

def RBF(x, beta, c):
    # x, c是(1,i_n) 维的
    # beta 是1维
    return np.exp(- beta * np.sqrt(np.dot((x-c), (x-c).T)))
X_trn = np.random.randint(0,2,(100,2))
y_trn = np.logical_xor(X_trn[:, 0], X_trn[:, 1])
