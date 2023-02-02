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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV


results = pd.read_csv("RFoptimize\RF遍历max_depth n_estimators.csv")

# show the first 5 rows
results["mean_test_score"] = abs(results["mean_test_score"])
xc = results["param_max_depth"].to_numpy()
ye = results["param_n_estimators"].to_numpy()
vm = abs(results["mean_test_score"].to_numpy())
print(xc)

results = pd.DataFrame(
    results, columns=["param_max_depth", "param_n_estimators", "mean_test_score"]
)

results = results.pivot(
    index="param_max_depth", columns="param_n_estimators", values="mean_test_score"
)
plot = sns.heatmap(results, cmap="viridis").invert_yaxis()
plt.xlabel("n estimators")
plt.ylabel("max depth")
plt.savefig("RFoptimize\drawing.svg")

