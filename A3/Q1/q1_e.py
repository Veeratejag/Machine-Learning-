from collections import deque

from sklearn.preprocessing import  OneHotEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV



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


rf_model = RandomForestClassifier(oob_score=True, random_state=42)

param_grid = {
    'n_estimators': range(50, 351, 100),
    'max_features': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    'min_samples_split': range(2, 11, 2)
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
y_train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
if X_val is not None and y_val is not None:
    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
else:
    val_accuracy = None

oob_accuracy = best_model.oob_score_
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Best parameters:", best_params)
print("Training set accuracy:", train_accuracy)
if val_accuracy is not None:
    print("Validation set accuracy:", val_accuracy)
print("Out-of-bag accuracy:", oob_accuracy)
print("Test set accuracy:", test_accuracy)



best_model=[]
best_Acc=0
for depth in {5, 10, 15, 20}:
    for min_samples in [2, 4, 6, 8]:
        for max_featue in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            print("Depth:",depth,"min_samples:",min_samples,"max_feature:",max_featue)
            clf = GradientBoostingClassifier(n_estimators=350, learning_rate=1.0,
                max_depth=depth,min_samples_split=min_samples ,max_features=max_featue,random_state=56).fit(X_train, y_train)
            # print(clf.score(X_train, y_train))
            # print(clf.score(X_test, y_test))
            val_acc=clf.score(X_val, y_val)
            print(val_acc)
            if(val_acc>best_Acc):
                best_Acc=val_acc
                best_model=[depth,min_samples,max_featue]
            print()
print("Best model:\n","Depth: ",best_model[0],"\nmin_samples: ",best_model[1],"\nmax_feature: ",best_model[2],"\nAccuracy: ",best_Acc)