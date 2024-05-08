import cv2,os
import numpy as np

x_test = []; y_test = []
classes = ['0','1','2','3','4','5']
dataset_path = 'svm/val/'
for x in classes:
    path = os.path.join(dataset_path, x)
    for y in os.listdir(path):
        image = cv2.imread(os.path.join(path, y))
        image = cv2.resize(image, (16,16))
        image = image.reshape(768)
        image=image/255.0
        x_test.append(image)
        y_test.append(int(x)+1)
x_test_tot = np.array(x_test)
y_test_tot = np.array(y_test)



datas = [];labelss = []
dataset_path = 'svm/train/'
for x in classes:
    data=[];labels=[]
    path = os.path.join(dataset_path, x)
    for y in os.listdir(path):
        image = cv2.imread(os.path.join(path, y))
        image = cv2.resize(image, (16,16))
        image = image.reshape(768)
        image=image/255.0
        data.append(image)
        labels.append(int(x)+1)
    datas.append(data)
    labelss.append(labels)

x_train_tot = np.array(datas)
y_train_tot = np.array(labelss)

