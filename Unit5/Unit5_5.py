'''引入数据，并对数据进行预处理'''

# step 1 引入数据
import pandas as pd
with open('D:\\Desktop\西瓜数据集3.0.csv', 'r', encoding='utf-8') as data_obj:
    df = pd.read_csv(data_obj)

# Step 2 对数据进行预处理
# 对离散属性进行独热编码，定性转为定量，使每一个特征的取值作为一个新的特征
# 增加特征量   Catagorical Variable -> Dummy Variable
# 两种方法：Dummy Encoding VS One Hot Encoding
# 相同点：将Catagorical Variable转换为定量特征
# 不同点：Dummy Variable将Catagorical Variable转为n-1个特征变量
# One Hot Encoding 将其转换为n个特征变量，但会存在哑变量陷阱问题
# pandas自带的get_dummies()函数，可以将数据集中的所有标称变量转为哑变量
# sklearn 中的OneHotEncoder 也可以实现标称变量转为哑变量(注意要将非数字型提前通过LabelEncoder编码为数字类型，再进行转换，且只能处理单列属性)
# pybrain中的_convertToOneOfMany()可以Converts the target classes to a 1-of-k representation, retaining the old targets as a field class.
                                     # 对target class独热编码，并且保留原target为字段类
'''
dataset = pd.get_dummies(df, columns=df.columns[:6])         # 将离散属性变为哑变量
dataset = pd.get_dummies(dataset, columns=[df.columns[8]])   # 将标签转为哑变量
                                                             # columns接受序列形式的对象，单个字符串不行
'''
dataset = pd.get_dummies(df)
pd.set_option('display.max_columns', 1000) # 把所有的列全部显示出来

X = dataset[dataset.columns[:-2]]
Y = dataset[dataset.columns[-2:]]
labels = dataset.columns._data[-2:]

# Step 3：将数据转换为SupervisedDataSet/ClassificationDtaSet对象
from pybrain.datasets import ClassificationDataSet
ds = ClassificationDataSet(19, 1, nb_classes=2, class_labels=labels)
for i in range(len(Y)):
    y = 0
    if Y['好瓜_是'][i] == 1:
        y = 1
        ds.appendLinked(X.ix[i], y)
ds.calculateStatistics()  # 返回一个类直方图？搞不懂在做什么

# Step 4: 分开测试集和训练集
testdata = ClassificationDataSet(19, 1, nb_classes=2, class_labels=labels)
testdata_temp, traindata_temp = ds.splitWithProportion(0.25)
for n in range(testdata_temp.getLength()):
    testdata.appendLinked(testdata_temp.getSample(n)[0],testdata_temp.getSample(n)[1])
print(testdata)
testdata._convertToOneOfMany()
print(testdata)
traindata = ClassificationDataSet(19, 1, nb_classes=2, class_labels=labels)
for n in range(traindata_temp.getLength()):
    traindata.appendLinked(traindata_temp.getSample(n)[0], traindata_temp.getSample(n)[1])
traindata._convertToOneOfMany()
'''
# 使用sklean的OneHotEncoder
# 缺点是只能单列进行操作，最后再复合，麻烦
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
a = LabelEncoder().fit_transform(df[df.columns[0]])
# dataset_One = OneHotEncoder.fit(df.values[])
# print(df['色泽']) # 单独的Series？
print(a)
aaa = OneHotEncoder(sparse=False).fit_transform(a.reshape(-1, 1))
print(aaa)
# 怎么复合暂时没写
'''

'''开始整神经网络'''

# Step 1 ：创建神经网络框架
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SoftmaxLayer
# 输入数据是 19维，输出是两维，隐层设置为5层
# 输出层使用Softmax激活，其他：学习率(learningrate=0.01),学习率衰减(lrdecay=1.0，每次训练一步学习率乘以)，
# 详细（verbose=False）动量因子(momentum=0最后时步的梯度？），权值衰减？(weightdecay=0.0)

n_h = 5
net = buildNetwork(19, n_h, 2, outclass=SoftmaxLayer)

# Step 2 : 构建前馈网络标准BP算法
from pybrain.supervised import BackpropTrainer
trainer_sd = BackpropTrainer(net, traindata)

# # 或者使用累积BP算法,训练次数50次
# trainer_ac = BackpropTrainer(net, traindata, batchlearning=True)
# trainer_ac.trainEpochs(50)
# err_train, err_valid = trainer_ac.trainUntilConvergence(maxEpochs=50)

for i in range(50): # 训练50次，每及测试结果次打印训练结果
    trainer_sd.trainEpochs(1) # 训练网络一次,

    # 引入训练误差和测试误差
    from pybrain.utilities import percentError
    trainresult = percentError(trainer_sd.testOnClassData(), traindata['class'])
    testresult = percentError(trainer_sd.testOnClassData(dataset=testdata), testdata['class'])
    # 打印错误率
    print('Epoch: %d', trainer_sd.totalepochs, 'train error: ', trainresult, 'test error: ', testresult)
