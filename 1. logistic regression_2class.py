#!/usr/bin/env python
# coding: utf-8

# In[2]:


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,auc,roc_curve
from sklearn.preprocessing import LabelBinarizer,StandardScaler

from sklearn import model_selection as cv
from sklearn.model_selection import cross_val_score

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn import metrics
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import roc_curve, auc

from itertools import cycle

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import itertools


# In[4]:


os.getcwd()


# In[5]:


df = pd.read_csv('./dataset/4.0 stroke2_joint.csv')


# In[22]:


C = df.iloc[:,1:16]
R = df.iloc[:,16]
D = df.iloc[:,17]
C_R = df.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
C_D = df.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17]]
D_R = df.iloc[:,[16,17]]
C_R_D = df.iloc[:,-5:]
Y = df.iloc[:,0]


# In[7]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# <font color=#0099ff  size=5 face="黑体">第一个数据集：C</font>

# In[8]:


x_train_c, x_test_c, y_train_c, y_test_c = train_test_split(C,Y, test_size=0.3, random_state = 3)


# In[9]:


print(Counter(x_train_c),Counter(x_test_c),Counter(y_train_c),Counter(y_test_c))


# In[10]:


lr_model_c = LogisticRegression(max_iter=1000)
lr_model_c.fit(x_train_c, y_train_c)


# In[11]:


c_test_pred = lr_model_c.predict(x_test_c)
c_test_proba = lr_model_c.predict_proba(x_test_c)
c_test_acc = accuracy_score(y_test_c, c_test_pred)  
print('ACC:{:.4f}'.format(c_test_acc ))
c_matrix = confusion_matrix(y_test_c, c_test_pred)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(c_matrix, classes=[0, 1], title='Confusion matrix')
plt.show()


# In[12]:


print(metrics.classification_report(y_test_c, c_test_pred,digits=4))


# In[13]:


fpr,tpr,threshold = roc_curve(y_test_c, c_test_proba[:,1]) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# <font color=#0099ff  size=5 face="黑体">第2个数据集：R</font>

# In[14]:


R_1 = np.array(R).reshape(-1,1)
x_train_r, x_test_r, y_train_r, y_test_r = train_test_split(R_1,Y, test_size=0.3, random_state = 3)


# In[15]:


lr_model_r = LogisticRegression(max_iter=1000)                                   #change
lr_model_r.fit(x_train_r, y_train_r)                                             #change
r_test_pred = lr_model_r.predict(x_test_r)                                       #change
r_test_proba = lr_model_r.predict_proba(x_test_r)                                #change
r_test_acc = accuracy_score(y_test_r, r_test_pred)                              #change
print('ACC:{:.4f}'.format(r_test_acc ))                                         #change
r_matrix = confusion_matrix(y_test_r, r_test_pred)                              #change
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(r_matrix, classes=[0, 1], title='Confusion matrix')            #change
plt.show()
print(metrics.classification_report(y_test_r, r_test_pred,digits=4))             #change
fpr,tpr,threshold = roc_curve(y_test_r, r_test_proba[:,1])                       #change
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# <font color=#0099ff  size=5 face="黑体">第3个数据集：D</font>

# In[16]:


D_1 = np.array(D).reshape(-1,1)                                   #change
x_train_d, x_test_d, y_train_d, y_test_d = train_test_split(D_1,Y, test_size=0.3, random_state = 3)      #change
lr_model_d = LogisticRegression(max_iter=1000)                                   #change
lr_model_d.fit(x_train_d, y_train_d)                                             #change
d_test_pred = lr_model_d.predict(x_test_d)                                       #change
d_test_proba = lr_model_d.predict_proba(x_test_d)                                #change
d_test_acc = accuracy_score(y_test_d, d_test_pred)                              #change
print('ACC:{:.4f}'.format(d_test_acc ))                                         #change
d_matrix = confusion_matrix(y_test_d, d_test_pred)                              #change
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(d_matrix, classes=[0, 1], title='Confusion matrix')            #change
plt.show()
print(metrics.classification_report(y_test_d, d_test_pred,digits=4))             #change
fpr,tpr,threshold = roc_curve(y_test_d, d_test_proba[:,1])                       #change
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# <font color=#0099ff  size=5 face="黑体">第4个数据集：C_R</font>

# In[17]:


# D_1 = np.array(D).reshape(-1,1)                                   #change
x_train_cr, x_test_cr, y_train_cr, y_test_cr = train_test_split(C_R,Y, test_size=0.3, random_state = 3)      #change
lr_model_cr = LogisticRegression(max_iter=1000)                                   #change
lr_model_cr.fit(x_train_cr, y_train_cr)                                             #change
cr_test_pred = lr_model_cr.predict(x_test_cr)                                       #change
cr_test_proba = lr_model_cr.predict_proba(x_test_cr)                                #change
cr_test_acc = accuracy_score(y_test_cr, cr_test_pred)                              #change
print('ACC:{:.4f}'.format(cr_test_acc ))                                         #change
cr_matrix = confusion_matrix(y_test_cr, cr_test_pred)                              #change
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cr_matrix, classes=[0, 1], title='Confusion matrix')            #change
plt.show()
print(metrics.classification_report(y_test_cr, cr_test_pred,digits=4))             #change
fpr,tpr,threshold = roc_curve(y_test_cr, cr_test_proba[:,1])                       #change
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# <font color=#0099ff  size=5 face="黑体">第5个数据集：C_D</font>

# In[18]:


# D_1 = np.array(D).reshape(-1,1)                                   #change
x_train_cd, x_test_cd, y_train_cd, y_test_cd = train_test_split(C_D,Y, test_size=0.3, random_state = 3)      #change
lr_model_cd = LogisticRegression(max_iter=1000)                                   #change
lr_model_cd.fit(x_train_cd, y_train_cd)                                             #change
cd_test_pred = lr_model_cd.predict(x_test_cd)                                       #change
cd_test_proba = lr_model_cd.predict_proba(x_test_cd)                                #change
cd_test_acc = accuracy_score(y_test_cd, cd_test_pred)                              #change
print('ACC:{:.4f}'.format(cd_test_acc ))                                         #change
cd_matrix = confusion_matrix(y_test_cd, cd_test_pred)                              #change
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cd_matrix, classes=[0, 1], title='Confusion matrix')            #change
plt.show()
print(metrics.classification_report(y_test_cd, cd_test_pred,digits=4))             #change
fpr,tpr,threshold = roc_curve(y_test_cd, cd_test_proba[:,1])                       #change
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# <font color=#0099ff  size=5 face="黑体">第6个数据集：D_R</font>

# In[23]:


# D_1 = np.array(D).reshape(-1,1)                                   #change
x_train_rd, x_test_rd, y_train_rd, y_test_rd = train_test_split(D_R,Y, test_size=0.3, random_state = 3)      #change
lr_model_rd = LogisticRegression(max_iter=1000)                                   #change
lr_model_rd.fit(x_train_rd, y_train_rd)                                             #change
rd_test_pred = lr_model_rd.predict(x_test_rd)                                       #change
rd_test_proba = lr_model_rd.predict_proba(x_test_rd)                                #change
rd_test_acc = accuracy_score(y_test_rd, rd_test_pred)                              #change
print('ACC:{:.4f}'.format(rd_test_acc ))                                         #change
rd_matrix = confusion_matrix(y_test_rd, rd_test_pred)                              #change
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(rd_matrix, classes=[0, 1], title='Confusion matrix')            #change
plt.show()
print(metrics.classification_report(y_test_rd, rd_test_pred,digits=4))             #change
fpr,tpr,threshold = roc_curve(y_test_rd, rd_test_proba[:,1])                       #change
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# <font color=#0099ff  size=5 face="黑体">第7个数据集：C_D_R</font>

# In[24]:


# D_1 = np.array(D).reshape(-1,1)                                   #change
x_train_crd, x_test_crd, y_train_crd, y_test_crd = train_test_split(C_R_D,Y, test_size=0.3, random_state = 3)      #change
lr_model_crd = LogisticRegression(max_iter=1000)                                   #change
lr_model_crd.fit(x_train_crd, y_train_crd)                                             #change
crd_test_pred = lr_model_crd.predict(x_test_crd)                                       #change
crd_test_proba = lr_model_crd.predict_proba(x_test_crd)                                #change
crd_test_acc = accuracy_score(y_test_crd, crd_test_pred)                              #change
print('ACC:{:.4f}'.format(crd_test_acc ))                                         #change
crd_matrix = confusion_matrix(y_test_crd, crd_test_pred)                              #change
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(crd_matrix, classes=[0, 1], title='Confusion matrix')            #change
plt.show()
print(metrics.classification_report(y_test_crd, crd_test_pred,digits=4))             #change
fpr,tpr,threshold = roc_curve(y_test_crd, crd_test_proba[:,1])                       #change
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[25]:


import matplotlib.pyplot as plt 
 
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 #有中文出现的情况，需要u'内容'


# 组合图

# In[32]:


fpr,tpr,threshold = roc_curve(y_test_c, c_test_proba[:,1]) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
plt.figure()
lw = 2
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='firebrick',
         lw=lw, label='C (area = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线

fpr,tpr,threshold = roc_curve(y_test_d, d_test_proba[:,1]) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='D (area = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线


fpr,tpr,threshold = roc_curve(y_test_r, r_test_proba[:,1]) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.plot(fpr, tpr, color='gold',
         lw=lw, label='R (area = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线

fpr,tpr,threshold = roc_curve(y_test_cr, cr_test_proba[:,1]) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.plot(fpr, tpr, color='yellowgreen',
         lw=lw, label='C+R (area = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线

fpr,tpr,threshold = roc_curve(y_test_cd, cd_test_proba[:,1]) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.plot(fpr, tpr, color='lime',
         lw=lw, label='C+D (area = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线

fpr,tpr,threshold = roc_curve(y_test_rd,rd_test_proba[:,1]) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.plot(fpr, tpr, color='blue',
         lw=lw, label='R+D (area = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线

fpr,tpr,threshold = roc_curve(y_test_crd, crd_test_proba[:,1]) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.plot(fpr, tpr, color='fuchsia',
         lw=lw, label='ALL (area = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig("ROC图.svg",dpi = 300, format = 'svg')

plt.show()


# In[ ]:




