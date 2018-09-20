import numpy as np
import matplotlib.pyplot as plt

#load data
dataset = np.loadtxt('D:\\Desktop\西瓜数据.csv',encoding='utf-8-sig',delimiter=',')
print(dataset)
X1 = dataset[:,1:3]
m, n = np.shape(X1)
X0 = np.ones((m,1))
X = np.hstack((X0,X1))
print(X)
Y = dataset[:,-1]
print(Y)


def likelihood_function(x, y, bate):
    """
定义似然函数
    :param x:
    :param y:
    :param bate:
    """
    ones = np.ones((m,1))
    # return -np.dot(np.dot(bate, x.T),y) + np.dot(np.log(1+np.exp(bate, x.T)),ones)
    return -np.dot(np.dot(bate,x.T), y) + np.dot(np.log(1+np.exp(np.dot(bate, x.T))), ones)


def gradient(x,y,bate):
    """
定义梯度
    :param x:
    :param y:
    :param bate:
    :return:
    """
    # return -np.dot(x.T, (y-(np.exp(np.dot(x,bate.T)))/(1+np.exp(np.dot(x,bate.T)))))
    return -np.dot(y.T - (np.exp(np.dot(bate, x.T)))/(1 + np.exp(np.dot(bate, x.T))), x)


def gradient_descent(x, y, alpha):
    """
定义梯度下降算法
    :param x:
    :param y:
    :param alpha:
    :return:
    """
    m, n = np.shape(x)
    bate = np.ones((1, n))
    grad = gradient(x, y, bate)
    while np.all(np.abs(gradient(x,y,bate)) > 1e-5):
        bate = bate - alpha*grad
        grad = gradient(x, y, bate)
    return bate


optimal = gradient_descent(X, Y, alpha=0.1).reshape(3,)
print('optimal: ', optimal,np.shape(optimal))
print('likelihood_function: ', likelihood_function(X,Y,optimal))
#draw scatter diggram to show the raw data
f1 = plt.figure()
plt.title('watermelon_3a')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.scatter(X[Y == 1, 1], X[Y == 1, 2], marker='o', color = 'r', label='good')
plt.scatter(X[Y == 0, 1], X[Y == 0, 2], marker='o', color = 'g', label='bad')
x = np.linspace(0.2, 0.8)
y = -optimal[0]/optimal[2]-(optimal[1]/optimal[2])*x
plt.plot(x,y)
plt.legend()
plt.show()
