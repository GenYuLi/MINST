from sklearn.metrics import accuracy_score
import time
import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt 
from tensorflow.python.client import device_lib
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, Conv2D
from keras.optimizers import rmsprop_v2
from keras.utils import np_utils
from keras import backend as K
from matplotlib.pyplot import imshow
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv("K:\AcadamicFiles\資料科學導論\project\\train.csv")
test = pd.read_csv("K:\AcadamicFiles\資料科學導論\project\\test.csv")
X_train = train.values[0:,1:]
Y_train = train.values[0:,0]
X_predict = test.values[0:,0:]
X_train = X_train.astype('float32')
X_train /= 255
X_predict = X_predict.astype('float32')
X_predict /= 255
Y_train = np_utils.to_categorical(Y_train, 10)

print(len(X_train))
print(len(X_train[1]))
print(len(Y_train))
print(Y_train[6])

K._get_available_gpus()
knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train, Y_train)
Y_predict = knn.predict(X_predict)
ans = pd.read_csv("K:\AcadamicFiles\資料科學導論\project\\sub.csv")
for i in range(28000):
    k=-1
    for j in range(10):
        if Y_predict[i][j] == 1:
            ans['Label'][i]=j
ans.to_csv('K:\AcadamicFiles\資料科學導論\project\\ans_sample_knn_n=7.csv',index=False)


predict_visual = test.values[0:300,0:]
print(len(predict_visual))
print(len(predict_visual[0]))
predict_visual = predict_visual.astype('float32')
predict_visual = predict_visual / 255.0
###資料視覺化###
pca = decomposition.PCA(n_components=3)
new_X = pca.fit_transform(predict_visual)
fig=plt.figure()
ax=Axes3D(fig)
X=new_X[:,0]
Y=new_X[:,1]
Z=new_X[:,2]
ax.set_xlim([min(X),max(X)])
ax.set_ylim([min(Y),max(Y)])
ax.set_zlim([min(Z),max(Z)])
for i in range(predict_visual.shape[0]):
    text=ans['Label'][i]
    ax.text(X[i],Y[i],Z[i],str(text),fontsize=8,bbox=dict(boxstyle='round',facecolor=plt.cm.Set1(text),alpha=0.7))
plt.show()