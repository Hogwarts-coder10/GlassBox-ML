import numpy as np

def train_test_split(X,y,test_size = 0.2,random_state = None,shuffle = True):
    """
    Splits arrays or matrices into random train and test subsets.
    This prevents the model from 'memorizing'.
    """

    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    # Always shuffle before splitting so we get a healthy mix of classes!
    if shuffle:
        np.random.shuffle(indices)
        
    # Calculate the exact row index where we need to slice the data
    split_idx = int(n_samples * (1 - test_size))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Slice the matrices
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test
