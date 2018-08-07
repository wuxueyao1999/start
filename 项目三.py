
# coding: utf-8

# In[12]:


from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.model_selection import train_test_split
def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]
path = u'D:/暑假集训/Python/项目/iris.data'  # 数据文件路径
data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
##print data



# In[13]:


x, y = np.split(data, (4,), axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)


# In[14]:


# clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(x_train, y_train.ravel())


# In[15]:


print u'训练集正确率：',clf.score(x_train, y_train)  # 精度
print u'测试集正确率：',clf.score(x_test, y_test)


# In[16]:


print 'decision_function:\n', clf.decision_function(x_train)#样本点到各类别的距离
print '\npredict:\n', clf.predict(x_train)# 用训练好的分类器去预测x_train数据的标签


# In[17]:


sepal_length = float(raw_input('please input the sepal length:'))
sepal_width = float(raw_input('please input the sepal width:'))
petal_length = float(raw_input('please input the petal length:')) 
petal_width = float(raw_input('please input the petal width:'))
a=[[sepal_length,sepal_width,petal_length,petal_width]]
print '\ncategory:\n', clf.predict(a)# 输出分类标号
b=clf.predict(a)
if b==0:
   print 'Iris-setosa'
elif b==1:
   print  'Iris-versicolor'
else:
   print  'Iris-virginica'
  

