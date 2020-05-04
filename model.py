import os
import pandas as pd
import math
import numpy as np

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
M = np.matrix(M)
target = M[:, 4]
data = M[:, :4]

# print(principal_components)
print(f"datashape = {data.shape}")

mlp = MLP(hidden_layers=(5, 5))
mlp.fit(data, target)
