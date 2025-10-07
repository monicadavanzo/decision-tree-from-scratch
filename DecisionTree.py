import numpy as np

class Node:
    def __init__(self,gini,num_samples,num_samples_per_class,predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class

        self.split_feature = None
        self.threshold = None
        self.left = None
        self.right = None

class DecisionTreeClassifier:
    def __init__(self,max_depth = None):
        self.max_depth = max_depth

        self.n_classes_ = None
        self.n_features_ = None
        self.sorted_idx_ = None
        self.tree_ = None
    
    def _gini(self,y):
        m = len(y)
        counts = np.array([np.sum(y==c) for c in range(self.n_classes_)])
        return 1 - np.sum((counts/m)**2)

    def _best_split(self,X,y,sorted_idx):
        node_samples = y[sorted_idx[0]]
        m = len(node_samples)
        if m <=1:
            return None,None
        best_feature = None
        best_threshold = None
        best_gain = -1e10
        num_parents = np.array([np.sum(node_samples==c) for c in range(self.n_classes_)])
        parent_gini = 1 - np.sum((num_parents/m)**2)

        for feat in range(self.n_features_):
            idx = sorted_idx[feat]
            X_sorted,y_sorted = X[idx,feat], y[idx]

            n_left = np.zeros(self.n_classes_)
            n_right = num_parents.copy()

            for i in range(1,m):
                c =  y_sorted[i-1]
                n_left[c] +=1
                n_right[c] -=1
                #Check for repeated values as this affects the gini for a given threshold
                if X_sorted[i-1] == X_sorted[i]:
                    continue
                gini_left = 1 - np.sum((n_left/i)**2)
                gini_right = 1- np.sum((n_right/(m-i))**2)
                gain = parent_gini - (i*gini_left + (m-i)*gini_right)/m

                if gain>best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = (X_sorted[i-1]+X_sorted[i])/2
        return best_feature, best_threshold
   
    def _grow_tree(self,X,y,sorted_idx,depth=0):
        node_samples = y[sorted_idx[0]]
        num_samples = len(node_samples)
        num_samples_per_class = [np.sum(node_samples == c) for c in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini = self._gini(node_samples),
            num_samples = num_samples,
            num_samples_per_class = num_samples_per_class,
            predicted_class = predicted_class,
        )
        if depth < self.max_depth or self.max_depth is None:
            best_feat, best_thr = self._best_split(X,y,sorted_idx)
            if best_feat is not None:
                idx_left = X[:,best_feat] <= best_thr
                left_sorted_idx = []
                right_sorted_idx = []
                for feature in range(self.n_features_):
                    idx_feature = sorted_idx[feature]
                    mask = idx_left[idx_feature]
                    left_sorted_idx.append(idx_feature[mask])
                    right_sorted_idx.append(idx_feature[~mask])
                left_sorted_idx = np.array(left_sorted_idx)
                right_sorted_idx = np.array(right_sorted_idx)

                node.split_feature = best_feat
                node.threshold = best_thr
                node.left = self._grow_tree(X,y,left_sorted_idx,depth+1)
                node.right = self._grow_tree(X,y,right_sorted_idx,depth+1)
        return node
                
    def fit(self,X,y):
        self.n_classes_=len(set(y))
        self.n_features_ = X.shape[1]
        # pre-sort
        self.sorted_idx = [np.argsort(X[:,feature]) for feature in range(self.n_features_)]
        self.tree_ = self._grow_tree(X,y,self.sorted_idx)
    
    def _predict_one(self,x,node):
        if node.left is None and node.right is None:
            return node.predicted_class
        if x[node.split_feature] <= node.threshold:
            return self._predict_one(x,node.left)
        else:
            return self._predict_one(x,node.right)
    
    def predict(self,X):
        return [self._predict_one(x,self.tree_) for x in X]