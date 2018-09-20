import numpy as np
X_trn = np.random.randint(0, 2, (100, 2))
y_trn = np.logical_xor(X_trn[:, 0], X_trn[:, 1])
centers = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# 构建网络
from RBF_net import *
net = Net()
net.Create_nn(4, centers, 0.05)

# 训练网络100次，看误差变化
err_list = []
for i in range(100):
    net.RBF_trainer(X_trn, y_trn)
    err = net.err
    err_list.append(err)
# 画出误差变化
import matplotlib.pyplot as plt
fig1 = plt.figure(1)
plt.plot(err_list)
plt.show()


