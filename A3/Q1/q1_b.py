from collections import deque

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

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

class DTNode:

    def __init__(self, depth, is_leaf = False, value = 0,threshold=None, column = None):

        #to split on column
        self.depth = depth
        self.threshold = threshold
        #add children afterwards
        self.children = None

        #if leaf then also need value
        self.is_leaf = is_leaf
        if(self.is_leaf):
            self.value = value
        
        if(not self.is_leaf):
            self.column = column


    def get_children(self, X):
        '''
        Args:
            X: A single example np array [num_features]
        Returns:
            child: A DTNode
        '''
        if self.is_leaf:
            return self
        elif types[self.column]=="cat":
            # print(self.column, X[self.column],len(self.children))
            if X[self.column] >= len(self.children):
                return self.children[-1]
            return self.children[int(X[self.column])]
        else:
            if X[self.column] <= self.threshold:
                return self.children[0]
            else:
                return self.children[1]
        
def entropy(X,y):
    if(len(y)==0):
        return 0
    entropy=0
    for i in np.unique(y):
        p=np.sum(y==i)/len(y)
        if p==1 or p==0:
            return 0
        entropy-=p*np.log2(p)
    return entropy

def information_gain(X,y,feature,iscat:bool):
    entropy_parent=entropy(X,y)
    final_entropy=0
    if not iscat:
        threshold=np.median(X[:,feature])
        left_X=X[X[:,feature]<=threshold]
        right_X=X[X[:,feature]>threshold]
        left_y=y[X[:,feature]<=threshold]
        right_y=y[X[:,feature]>threshold]
        p=len(left_y)/len(y)
        final_entropy=p*entropy(left_X,left_y)+(1-p)*entropy(right_X,right_y)
    else:
        children = []
        values=np.unique(X[:,feature])
        for i in values:
            child_X = X[X[:, feature] == i]
            child_y = y[X[:, feature] == i]
            children.append((child_X, child_y))
        
        for child_X, child_y in children:
            final_entropy += len(child_y) / len(y) * entropy(child_X, child_y)
    return entropy_parent-final_entropy
def split(X,y,types):
    best_feature=None
    best_ig=-1
    best_threshold=None
    for i in range(len(types)):
        ig=information_gain(X,y,i,types[i]=='cat')
        if(ig>best_ig):
            best_ig=ig
            best_feature=i
            if(types[i]=='cat'):
                best_threshold=None
            else:
                best_threshold=np.median(X[:,i])
    return best_feature,best_threshold

class DTTree:

    def __init__(self):
        #Tree root should be DTNode
        self.root = None

    def fit(self, X, y, types, max_depth = 10):
        '''
        Makes decision tree
        Args:
            X: numpy array of data [num_samples, num_features]
            y: numpy array of classes [num_samples, 1]
            types: list of [num_features] with types as: cat, cont
                eg: if num_features = 4, and last 2 features are continious then
                    types = ['cat','cat','cont','cont']
            max_depth: maximum depth of tree
        Returns:
            None
        '''
        self.root = self.grow_tree(X, y, types, max_depth, 0)
        #TODO
    def grow_tree(self, X, y, types, max_depth, depth):
        if depth == max_depth or len(np.unique(y)) == 1:
            return DTNode(depth, is_leaf = True, value = np.bincount(y.flatten()).argmax())
        else:
            best_col, best_split =split(X, y, types)
            if best_col is None:
                print("best col is none", depth)
                return DTNode(depth, is_leaf = True, value = np.bincount(y.flatten()).argmax())
            if np.unique(X[:,best_col]).shape[0] ==1:
                return DTNode(depth, is_leaf = True, value = np.bincount(y.flatten()).argmax())
            else:
                node = DTNode(depth, is_leaf = False,threshold=best_split, column = best_col)
                if types[best_col] == "cat":
                    node.children = []
                    for i in np.unique(X[:,best_col]):
                        child_X = X[X[:,best_col]==i]
                        child_y = y[X[:,best_col]==i]
                        child = self.grow_tree(child_X, child_y, types, max_depth, depth+1)
                        node.children.append(child)
                    
                else:
                    left_X = X[X[:,best_col]<=best_split]
                    left_y = y[X[:,best_col]<=best_split]
                    right_X = X[X[:,best_col]>best_split]
                    right_y = y[X[:,best_col]>best_split]
                    left_subtree = self.grow_tree(left_X, left_y, types, max_depth, depth+1)
                    right_subtree = self.grow_tree(right_X, right_y, types, max_depth, depth+1)
                    node.children = [left_subtree, right_subtree]
                return node
    
    def __call__(self, X):
        '''
        Predicted classes for X
        Args:
            X: numpy array of data [num_samples, num_features]
        Returns:
            y: [num_samples, 1] predicted classes
        '''
        y_pred = []
        for x in X:
            y_pred.append(self.predict(x))
        return y_pred
    def predict(self, x):
        node = self.root
        while node.is_leaf == False:
            node = node.get_children(x)
        return node.value
    def post_prune(self, X_val, y_val):
        node = self.root
        if not node:
            return node
        bfs = deque([node])
        while bfs:
            curr = bfs.popleft()
            
        pass


for max_depth in [15, 25, 35, 45]:
    print("Max Depth: ",max_depth)
    tree = DTTree()
    tree.fit(X_train,y_train,types, max_depth = max_depth)
    y_pred = tree(X_train)
    print("Training Accuracy: ",accuracy_score(y_train, y_pred))
    y_pred = tree(X_test)
    print("Testing Accuracy: ",accuracy_score(y_test, y_pred))
    print()