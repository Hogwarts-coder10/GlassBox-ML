import numpy as np
import matplotlib.pyplot as plt

from glassboxml.models import DecisionTreeClassifier
from glassboxml.datasets import make_blobs

X,y = make_blobs(n_samples = 500,n_features = 2, centers = 3,cluster_std = 1.5,random_state=42)
tree = DecisionTreeClassifier(max_depth = 3, criterion='entropy')
tree.fit(X,y)

print(tree.explain())
