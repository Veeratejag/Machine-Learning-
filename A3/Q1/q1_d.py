from collections import deque

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def get_np_array(file_name):
    label_encoder = None 
    data = pd.read_csv(file_name)
    
    need_label_encoding = ['team','host','opp','month', 'day_match']
    if(label_encoder is None):
        label_encoder = OneHotEncoder(sparse_output = False)
        label_encoder.fit(data[need_label_encoding])
    data_1 = pd.DataFrame(label_encoder.transform(data[need_label_encoding]), columns = label_encoder.get_feature_names_out())
    # print(data_1.shape)
    #merge the two dataframes
    dont_need_label_encoding =  ["year","toss","bat_first","format" ,"fow","score" ,"rpo" ,"result"]
    data_2 = data[dont_need_label_encoding]
    final_data = pd.concat([data_1, data_2], axis=1)
    
    X = final_data.iloc[:,:-1]
    y = final_data.iloc[:,-1:]
    return X.to_numpy(), y.to_numpy()

X_train,y_train = get_np_array('train.csv')
X_test, y_test = get_np_array("test.csv")

#only needed in part (c)
X_val, y_val = get_np_array("val.csv")

types = ['cat','cat','cat',"cat","cat","cont","cat","cat","cat" ,"cont","cont" ,"cont" ]
while(len(types) != X_train.shape[1]):
    types = ['cat'] + types


d={}
for depth in {15, 25, 35, 45}:
    tree = DecisionTreeClassifier(max_depth=depth, criterion="entropy")
    tree.fit(X_test, y_test)
    print("Depth:", depth)
    
    # Calculate accuracy for the training set
    y_train_pred = tree.predict(X_test)
    train_accuracy = accuracy_score(y_test, y_train_pred)
    print("Train Accuracy:", train_accuracy)

    # Calculate accuracy for the test set
    y_test_pred = tree.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test Accuracy:", test_accuracy)

    # Calculate accuracy for the validation set (if defined)
    if 'X_val' in locals() and 'y_val' in locals():
        y_val_pred = tree.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print("Val Accuracy:", val_accuracy)
        d[depth]=val_accuracy
    print()
final_depth=max(d,key=d.get)
print("Final Depth:",final_depth)


d={}
for ccp in {0.001, 0.01, 0.1, 0.2}:
    tree = DecisionTreeClassifier( criterion="entropy",ccp_alpha=ccp)
    tree.fit(X_test, y_test)
    print("CCP:", ccp)
    
    # Calculate accuracy for the training set
    y_train_pred = tree.predict(X_test)
    train_accuracy = accuracy_score(y_test, y_train_pred)
    print("Train Accuracy:", train_accuracy)

    # Calculate accuracy for the test set
    y_test_pred = tree.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test Accuracy:", test_accuracy)

    # Calculate accuracy for the validation set (if defined)
    if 'X_val' in locals() and 'y_val' in locals():
        y_val_pred = tree.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print("Val Accuracy:", val_accuracy)
        d[ccp]=val_accuracy
    print()
final_ccp=max(d,key=d.get)
print("Final CCP:",final_ccp)