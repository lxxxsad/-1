#实验一
clc;clear;close all;
load('abalone_data.mat')
n=size(data,2);
x=data(:,1:n-1); 
y=data(:,n);      
xm=mean(x);    
xs=std(x);         
ym=mean(y);
x_o=zscore(x);
k1=[1:0.1:10];
for k=1:length(k1)
%用标准化后的进行多元线性回归
lamda=k1(k);
xishu=(x_o'*x_o+lamda*eye(size(x_o,2)))^(-1)*x_o'*(y-ym);
%还原为原自变量的系数
xishu1=[ym-sum(xishu.*xm'./xs');xishu./xs'];
y_n1=xishu1(1)+x*xishu1(2:end);
wucha(k)=sum(abs(y_n1-y)./y)/length(y);
end
plot(wucha,'LineWidth',2)
ylabel('相对误差')
xlabel('岭参数')
set(gca,'XTick',1:10:100)
set(gca,'XtickLabel',1:1:10)

#实验二
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
data = load_iris() 
iris_target = data.target 
iris_features = pd.DataFrame(data=data.data, columns=data.feature_names) #利用Pandas转化为DataFrame格式
## 划分为训练集和测试集
from sklearn.model_selection import train_test_split
## 选择其类别为0和1的样本 （不包括类别为2的样本）
iris_features_part = iris_features.iloc[:100]
iris_target_part = iris_target[:100]
## 训练集测试集7/3分
x_train, x_test, y_train, y_test = train_test_split(iris_features_part, iris_target_part, test_size = 0.3, random_state = 2020)
## 从sklearn中导入逻辑回归模型
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='lbfgs')
# 训练模型
clf.fit(x_train, y_train)
## 预测模型
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)

#实验三
import numpy as np
import math
from scipy.io import loadmat
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
# 数据准备
minist_path = "/home/aistudio/datasets/lung.mat"
lung_path = "/home/aistudio/datasets/MNIST.mat"
yale_path = "/home/aistudio/datasets/Yale.mat"
KERNEL = ['linear', 'rbf', 'laplace']  # 基于线性核'linear'，基于高斯核'rbf',基于拉普拉斯核 'laplace'
# 加载数据
def create_data(path):
    data = loadmat(path)
    data_x = data["X"]
    data_y = data["Y"][:, 0]
    data_y -= 1
    Data = np.array(data_x)
    Label = np.array(data_y)
    return Data, Label
def laplace(X1, X2):
    K = np.zeros((len(X1), len(X2)), dtype=np.float)
    for i in range(len(X1)):
        for j in range(len(X2)):
            K[i][j] = math.exp(-math.sqrt(np.dot(X1[i] - X2[j], (X1[i] - X2[j]).T))/2)
    return K
def classify(path, kernel):
    X, y = create_data(path)
    train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.2, random_state=1000)
    # 训练svm分类器
    classifier = svm.SVC(C=2, kernel=kernel, gamma=10, decision_function_shape='ovr') # ovr:一对多策略
    classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先
    # w = classifier.coef_[0]  # 获取w
    # a = -w[0] / w[1]  # 斜率
    # 计算svc分类器的准确率
    print("训练集：", classifier.score(train_data, train_label))
    print("测试集：", classifier.score(test_data, test_label))
if __name__ == '__main__':
    print('MINIST数据集: ')
    print('线性: ')         
    classify(minist_path, KERNEL[0])
    print('高斯核: ')
    classify(minist_path, KERNEL[1])
    print('拉普拉斯核:')
classify(minist_path, KERNEL[2])
if __name__ == '__main__':
    print('lung数据集: ')
    print('线性: ')         
    classify(lung_path, KERNEL[0])
    print('高斯核: ')
    classify(lung_path, KERNEL[1])
    print('拉普拉斯核:')
    classify(lung_path, KERNEL[2])

    #实验四
    # 定义数据集
transform=paddle.vision.transforms.Normalize(mean=[127.5],std=[127.5])
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
### 搭建全连接网络
class mulitlayer(paddle.nn.Layer):
    def __init__(self):
        super(mulitlayer,self).__init__()
        self.flatten=paddle.nn.Flatten()
        self.linear1=paddle.nn.Linear(28*28,128)
        self.linear2=paddle.nn.Linear(128,64)
        self.outlayer=paddle.nn.Linear(64,10)
        self.relu=paddle.nn.ReLU()
    def forward(self,x):
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.relu(x)
        x=self.linear2(x)
        x=self.relu(x)
        x=self.outlayer(x)
        return x
### 开始训练
model=paddle.Model(mulitlayer())
model.prepare(
      optimizer=paddle.optimizer.Adam(learning_rate=0.01,parameters=model.parameters()),
      loss=paddle.nn.CrossEntropyLoss(),
      metrics=paddle.metric.Accuracy()
)
model.fit(
      train_data=train_dataset,
      eval_data=test_dataset,
      batch_size=64,
      epochs=1,
      verbose=1
)

    

