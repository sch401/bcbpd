from sklearn.preprocessing import normalize
import pandas as pd
from matplotlib.ticker import MultipleLocator
from matplotlib import pyplot as plt
from sklearn.model_selection import (
    cross_validate,
    cross_val_predict,
)
from matplotlib import ticker
import numpy as np
import seaborn as sns

plt.rcParams["font.size"] = 18
fig = plt.figure()
# from sklearn.svm import KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor

results = pd.read_csv("KNNoptimize\knn遍历neighbourp.csv")

# show the first 5 rows

xc = results["param_n_neighbors"].to_numpy()
ye = results["param_p"].to_numpy()
results["mean_test_score"] = abs(results["mean_test_score"])
vm = abs(results["mean_test_score"].to_numpy())
print(vm)
results = pd.DataFrame(
    results, columns=["param_n_neighbors", "param_p", "mean_test_score"]
)

results = results.pivot(
    index="param_n_neighbors", columns="param_p", values="mean_test_score"
)
plot = sns.heatmap(results, cmap="viridis").invert_yaxis()
plt.xlabel("p")
plt.ylabel("n neighbour")
plt.savefig("KNNoptimize\drawing.svg")
