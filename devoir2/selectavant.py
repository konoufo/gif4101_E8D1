import numpy as np
from sklearn.utils import check_X_y
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import safe_mask
from warnings import warn


class SelAvant:

    def __init__(self, estimator, n_features_to_select = None, step =1 ):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
       

    
    def fit(self, X, y):
        
        X, y = check_X_y(X, y)
        # Initialisation
        
        n_features = X.shape[1]
        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        else:
            n_features_to_select = self.n_features_to_select
        
        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")
        
        indices = np.arange(n_features)
        support_ = np.zeros(n_features, dtype=np.bool)
        support_ = np.concatenate([indices,support_])
        support_ = np.reshape(support_,(2,n_features)).T
            
        # Selection
        while np.sum(support_[:,1]) < n_features_to_select:
            
            features = support_[support_[:,1]==1,0]
            remainder = support_[support_[:,1]==0,0]
            perf_tested = []
            # selected features
            for i in remainder:
                features = support_[support_[:,1]==1,0]
                features = np.hstack([features, i ])
                #Evaluate the selected features
                estimator = clone(self.estimator)   
                estimator.fit(X[:, features], y)
                perf = estimator.score(X[:,features],y)
                perf_tested = np.hstack([perf_tested,perf])
            
            max_feat = perf_tested.argmax(axis=0) #index de feature selectionnée dans perf_tested
            select_feat =  remainder[max_feat]   #feature selected
            support_[support_[:,0]==select_feat, 1] = 1  #réinitialiser la matrice de sélection
            
            
        # Set final attributes
        features = support_[support_[:,1]==1,0]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, features], y)
        self.support_ = support_
                
        return self

    
    def _get_support_mask(self):
        check_is_fitted(self, 'support_')
        return self.support_

    def get_support(self, indices=False):
        
        mask = self._get_support_mask()
        if indices is False:
            mask = mask[:,1]
            return mask 
        else:
            ind = mask[mask[:,1]==1,0]
            return ind
        
        
    def transform(self, X):
           
          mask = self.get_support(indices=True)
          if not mask.any():
              warn("No features were selected: either the data is"
                   " too noisy or the selection test too strict.",
                       UserWarning)
              return np.empty(0).reshape((X.shape[0], 0))
         
              
          return X[:, mask[:]]
