import os
import pandas as pd
import math
import numpy as np
from scipy.cluster.vq import whiten

from MLP import MLP

dirname = os.path.dirname(__file__)
pathname = os.path.join(dirname, "training_data.csv")

df = pd.read_csv(pathname)
# Remove all points which are censored at 50.0 MEDV because the
# censoring takes their value away.
df = df[df.MEDV != 50.0]
df = df.drop(columns=["ZN", "INDUS", "CHAS", "NOX", "AGE", "DIS", "RAD", "TAX", "B"])

"""splitting data & target"""
M = df.to_numpy()
M = np.array(M)
target = M[:, 4]
data = M[:, :4]
# data = whiten(data)

# print(data)
# print(target)

# print(principal_components)
print(f"datashape = {data.shape}")

while True:
    mlp = MLP(hidden_layers=(50, 60, 40), iterations=10000)
    mlp.fit(data, target)

    print(f"Result: {mlp.predict(data)}")
    print(f"Loss: {np.mean(np.square(target - mlp.predict(data)))}")

# print(np.array([target]).T - mlp.predict(data))

# print(mlp.weights)
# print(mlp.bias)
