import numpy as np

class Net():
    def __init__(self, i_n=0, h_n=0, o_n=0, h_acf='Sigmoid', o_acf='Sigmoid'):
        # 初始化各层的层数
        self.i_n = i_n
        self.h_n = h_n
        self.o_n = o_n
        # 初始化各层的激活函数
        self.acf = {'Sigmoid': Sigmoid,
                    'Tanh': Tanh}
        self.h_acf = self.acf[h_acf]
        self.o_acf = self.acf[o_acf]
        self.d_acf = {Sigmoid: SigmodDerivate,
                      Tanh: TanhDerivate}
        # 初始化各层的输出形式
        self.i_v = np.zeros(self.i_n)
        self.h_v = np.zeros(self.h_n)
        self.o_v = np.zeros(self.o_n)
        # 初始化各层的权重阀值
        self.ih_w = np.random.uniform(-np.sqrt(6)/(self.i_n + self.h_n), np.sqrt(6)/(self.i_n + self.h_n), size=(self.i_n, self.h_n))
        self.ho_w = np.random.uniform(-np.sqrt(6)/(self.h_n + self.o_n), np.sqrt(6)/(self.h_n + self.o_n), size=(self.h_n, self.o_n))
        self.h_t = np.random.rand(1, self.h_n)
        self.o_t = np.random.rand(1, self.o_n)
        # 初始化训练次数
        self.totalEpoches = 0
        # 初始化误差
        self.err = 0.0
        # 初始化学习率
        self.lr_h = 0.01
        self.lr_o = 0.01
        # 初始化梯度
        # self.o_grid = None
        # self.h_grid = None

    def BP_trainer(self, inputset, outputset, lr_h, lr_o, batchlearning=False):
        self.i_v = inputset
        self.labelset = outputset
        self.batchlearning = batchlearning
        self.err = 0.0
        self.lr_h = lr_h
        self.lr_o = lr_o
        if self.batchlearning is False:  # 标准BP算法
            err_sum = 0.0
            for i in range(len(self.i_v)):
                # 这里每一层都要调用激活函数了
                x = self.i_v[i].reshape(1, self.i_n)
                self.h_v = self.h_acf(x.dot(self.ih_w) - self.h_t)
                self.o_v = self.o_acf(self.h_v.dot(self.ho_w) - self.o_t)
                # 激活函数导数
                self.hd_acf = self.d_acf[self.h_acf]
                self.od_acf = self.d_acf[self.o_acf]
                # 误差
                err_sum += np.dot((self.o_v - self.labelset[i]), (self.o_v - self.labelset[i]).T)/2
                # 更新公式(ps:绕的晕晕的)
                # 计算梯度
                self.o_grid = (self.labelset[i]-self.o_v) * self.od_acf(self.o_v)
                self.h_grid = self.hd_acf(self.h_v) * np.dot(((self.labelset[i]-self.o_v)*self.od_acf(self.o_v)),(self.ho_w.T))
                # 更新公式
                self.ih_w += self.lr_h * np.dot(x.T, self.h_grid)
                self.ho_w += self.lr_o * (self.h_v.T .dot(self.o_grid))
                self.h_t -= self.lr_h * self.h_grid
                self.o_t -= self.lr_o * self.o_grid
            self.totalEpoches += 1
            self.err = (err_sum/len(self.labelset)).reshape(-1)
        else:
            # 累积BP算法
            self.h_v = self.h_acf(self.i_v.dot(self.ih_w) - self.h_t)
            self.o_v = self.o_acf(self.h_v.dot(self.ho_w) - self.o_t)
            ones_vec = np.ones((1, len(self.i_v)))
            self.err = np.dot(ones_vec, np.dot((self.o_v - self.labelset)**2, np.ones((self.o_n, 1)))).reshape(-1)/(2*len(self.i_v))
            self.hd_acf = self.d_acf[self.h_acf]
            self.od_acf = self.d_acf[self.o_acf]
            # 参数更新公式
            self.o_grid = np.dot(ones_vec, (self.labelset - self.o_v) * self.od_acf(self.o_v)) / len(self.labelset)
            self.h_grid = (np.dot(ones_vec, self.hd_acf(self.h_v))/len(self.labelset)) * np.dot(self.o_grid, self.ho_w.T)
            self.ih_w += self.lr_h * np.dot(np.dot(ones_vec, self.i_v).T/len(self.labelset), self.h_grid)
            self.ho_w += self.lr_o * np.dot(np.dot(ones_vec, self.h_v).T/len(self.labelset), self.o_grid)
            self.h_t -= self.lr_h * self.h_grid
            self.o_t -= self.lr_o * self.o_grid
            self.totalEpoches += 1

    def BP_trainerDynamic(self,inputset, outputset, alpha):
        """
         第t次的学习率为：lr(t) = lr(t-1) * (2**lamda)
                          lamda = sign(grid(t) * grid(t-1))
         思想：若两次的梯度方向相同则步长（学习率）加快(翻倍)，否则减慢（减半）
         第t次的权重梯度为： dialt_w(t) = lr(t) * grid(t) * input + alpha * dialt_w(t-1)
         第t次的阀值梯度为： dialt_td(t) = lr(t) * grid(t) + alpha * dialt_td(t-1)
        :param inputset:
        :param outputset:
        :param alpha:
        """
        self.i_v = inputset
        self.labelset = outputset
        self.err = 0.0
        err_sum = 0.0
        for i in range(len(self.i_v)):
            # self.i_v = inputset[i].reshape(1, self.i_n)
            # 这里每一层都要调用激活函数了
            x = self.i_v[i].reshape(1, self.i_n)
            self.h_v = self.h_acf(x.dot(self.ih_w) - self.h_t)
            self.o_v = self.o_acf(self.h_v.dot(self.ho_w) - self.o_t)
            # 激活函数导数
            self.hd_acf = self.d_acf[self.h_acf]
            self.od_acf = self.d_acf[self.o_acf]
            # 误差
            err_sum += np.dot((self.o_v - self.labelset[i]), (self.o_v - self.labelset[i]).T) / 2

            # 计算梯度
            # 对于初始第一步，梯度为None，无法跟据上一步求，所以第一步采取固定学习率求法
            if self.o_grid == None:
                self.o_grid = (self.labelset[i] - self.o_v) * self.od_acf(self.o_v)
                self.h_grid = self.hd_acf(self.h_v) * np.dot((self.labelset[i] - self.o_v) * self.od_acf(self.o_v),
                                                             self.ho_w.T)
                self.ih_w += self.lr_h * np.dot(x.T, self.h_grid)
                self.ho_w += self.lr_o * self.h_v.T.dot(self.o_grid)
                self.h_t -= self.lr_h * self.h_grid
                self.o_t -= self.lr_o * self.o_grid
            else:
                # 求出动态学习率
                o_grid_temp = (self.labelset[i] - self.o_v) * self.od_acf(self.o_v)
                lamda_o = np.sign(self.o_grid * o_grid_temp)
                h_grid_temp = self.hd_acf(self.h_v) * np.dot((self.labelset[i] - self.o_v) * self.od_acf(self.o_v),
                                                         self.ho_w.T)
                lamda_h = np.sign(self.h_grid * h_grid_temp)

                lr_o_temp = self.lr_o * (2**lamda_o)
                if lr_o_temp > 0.5:
                    self.lr_o = 0.5
                else:
                    self.lr_o = 0.005 if lr_o_temp < 0.005 else lr_o_temp
                # 控制学习率防止过大或者过小
                lr_h_temp = self.lr_h * (2**lamda_h)
                if lr_h_temp > 0.5:
                    self.lr_h = 0.5
                else:
                    self.lr_h = 0.005 if lr_h_temp < 0.005 else lr_h_temp
                # 梯度
                self.o_grid = o_grid_temp
                self.h_grid = h_grid_temp
                # 更新
                self.ih_w += (self.lr_h * np.dot(x.T, self.h_grid) + alpha * self.delta_ih_w)
                self.delta_ih_w = self.lr_h * np.dot(x.T, self.h_grid)
                self.ho_w += (self.lr_o * self.h_v.T.dot(self.o_grid) + alpha * self.delta_oh_w)
                self.delta_oh_w = self.lr_o * self.h_v.T.dot(self.o_grid)
                self.h_t -= self.lr_h * self.h_grid
                self.o_t -= self.lr_o * self.o_grid
        self.totalEpoches += 1
        self.err = (err_sum / len(self.labelset)).reshape(-1)


    def train_epoch(self, num):
        # if num == 1:
        #     return self
        # else:
        for i in range(num):
            self.BP_trainer(inputset=self.i_v, outputset=self.labelset, lr_h=self.lr_h, lr_o=self.lr_o, batchlearning=self.batchlearning)

    def predict_label(self, x):
        # 简单的二分类预测
        self.i_v = x
        self.h_v = self.h_acf(self.i_v.dot(self.ih_w) - self.h_t)
        self.o_v = self.o_acf(self.h_v.dot(self.ho_w) - self.o_t)
        y = []
        for i in self.o_v:
            if i[0] > 0.5:
                y.append(1)
            else:
                y.append(0)
        return np.array(y)

    def predict_acc(self, y):
        acc = 0.0
        for i in range(len(y)):
            if self.o_v[i] == y[i]:
                acc += 1
        return acc/len(y)

def Sigmoid(x):
    from numpy import exp
    return 1.0 / (1.0 + exp(-x))

def SigmodDerivate(y):
    return y * (1-y)

def Tanh(x):
    from numpy import tanh
    return tanh(x)

def TanhDerivate(y):
    return 1 - y * y
