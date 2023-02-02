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


results = pd.read_csv("ANNoptimize\ANN遍历firstcecond.csv")

results = results.pivot(
    index="first layer", columns="second layer", values="MAPE in test"
)
plot = sns.heatmap(results, cmap="viridis").invert_yaxis()
print(results)
plt.ylabel("hidden node number in first layer")
 
plt.xlabel("hidden node number in second layer")
plt.savefig("ANNoptimize\ANN遍历firstcecond.svg")


# # show the first 5 rows

# xc = results["param_C"].to_numpy()
# ye = results["param_epsilon"].to_numpy()
# zg = results["param_gamma"].to_numpy()
# vm = results["mean_test_score"].to_numpy()
# min_v = min(vm)


# max_v = max(vm)
# color = [
#     plt.get_cmap("seismic", 640)(int(float(i - min_v) / (max_v - min_v) * 640))
#     for i in vm
# ]


# fig = plt.figure()
# ax1 = fig.add_subplot(111, projection="3d")
# # ax1.text(4.9, 0.0002, 1.8,s="123",fontsize=12)
# plt.set_cmap(plt.get_cmap("seismic", 640))
# im = ax1.scatter(xc, ye, zg, c=color)
# # ax1.legend()

# fig.colorbar(
#     im,
#     format=matplotlib.ticker.FuncFormatter(
#         lambda x, pos: float("%.3f" % (x * (max_v - min_v) + min_v))
#     ),
# )
# # fig.colorbar(ax1)
# ax1.set_xlabel("C")
# ax1.set_ylabel("epsilon")
# ax1.set_zlabel("gamma")

# fig.savefig("SVR CV predicted heatmap.svg")
