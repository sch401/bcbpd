from sklearn.preprocessing import normalize
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import (
    cross_validate,
    cross_val_predict,
)
from plotly.offline import iplot
import plotly.graph_objs as go

from sklearn.model_selection import GridSearchCV
import numpy as np
import seaborn as sns
from sklearn.svm import SVR

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
# plt.rcParams["font.size"] =
Sourcedata = pd.read_excel("dataset_origin.xlsx", sheet_name="Sheet2")
data_scaled = normalize(Sourcedata)

X = data_scaled[:, :-1]
y = data_scaled[:, -1]


# ScoreAll = []
# for i in np.arange(4, 5, 0.01):
#     svr = SVR(kernel="rbf", C=i)
#     score = cross_val_score(svr, X, y, cv=10, scoring="neg_mean_absolute_percentage_error").mean()
#     ScoreAll.append([i, score])
# ScoreAll = np.array(ScoreAll)
# plt.figure()
# plt.plot(ScoreAll[:, 0], ScoreAll[:, 1])
# plt.show()
# C = 4.3

# ScoreAll = []
# for i in range(7, 20):
#     svr = SVR(kernel="rbf", C=4.3, gamma=i / 10)
#     score = cross_val_score(svr, X, y, cv=10, scoring="neg_mean_absolute_percentage_error").mean()
#     ScoreAll.append([i, score])
# ScoreAll = np.array(ScoreAll)
# plt.figure()
# plt.plot(ScoreAll[:, 0], ScoreAll[:, 1])
# plt.show()
#  gamma = 1.6

# ScoreAll = []
# for i in np.logspace(4, 2, 20, endpoint=True, base=0.1):
#     svr = SVR(kernel="rbf", C=4.3, gamma=1.6,epsilon = i)
#     score = cross_val_score(svr, X, y, cv=10, scoring="neg_mean_absolute_percentage_error").mean()
#     ScoreAll.append([i, score])
# ScoreAll = np.array(ScoreAll)
# plt.figure()
# plt.plot(ScoreAll[:, 0], ScoreAll[:, 1])
# plt.show()
#  gamma = 0.00085



C = list(np.arange(1, 10, 0.25))
gamma = list(np.arange(1.6, 2.6, 0.2))
epsilon = list(np.logspace(5, 2, 10, endpoint=True, base=0.1))
print(epsilon)
# # kernel = ["rbf", "poly"]
param_grid = [
    {
        "C": C,
        "gamma": gamma,
        "epsilon": epsilon,
        # "kernel": kernel,
    },
]

svr = SVR(kernel="rbf")
grid_search = GridSearchCV(
    svr, param_grid, cv=10, scoring="neg_mean_absolute_percentage_error"
)
grid_search.fit(X, y)
print(grid_search.best_params_)
print(grid_search.best_score_)
results = pd.DataFrame(grid_search.cv_results_)
results.to_csv("svmoptimize\svm遍历C e gamma.csv")
# show the first 5 rows

