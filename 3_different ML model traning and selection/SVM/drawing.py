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
import matplotlib

results = pd.read_csv("svmoptimize\svm遍历C e gamma.csv")

# show the first 5 rows

xc = results["param_C"].to_numpy()
ye = results["param_epsilon"].to_numpy()
zg = results["param_gamma"].to_numpy()
vm = abs(results["mean_test_score"].to_numpy())
min_v = min(vm)
max_v = max(vm)
color = [
    plt.get_cmap("viridis", 640)(int(float(i - min_v) / (max_v - min_v) * 640))
    for i in vm
]


fig = plt.figure()
ax1 = fig.add_subplot(111, projection="3d")
# ax1.text(4.9, 0.0002, 1.8,s="123",fontsize=12)
plt.set_cmap(plt.get_cmap("viridis", 640))
im = ax1.scatter(xc, ye, zg, c=color)
im2 = ax1.scatter(4.9, 2.154e-3, 1.8, c="red")
# ax1.legend()

fig.colorbar(
    im,
    location="right",
    fraction=0.03,
    pad=0.10,
    format=matplotlib.ticker.FuncFormatter(
        lambda x, pos: float("%.3f" % abs(x * (max_v - min_v) + min_v))
    ),
)
# fig.colorbar(ax1)
ax1.set_xlabel("C")
ax1.set_ylabel("epsilon")
ax1.set_zlabel("gamma")

plt.show()


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


# C = list(np.arange(1, 6, 0.25))
# gamma = list(np.arange(1.6, 2.6, 0.2))
# epsilon = list(np.logspace(5, 2, 10, endpoint=True, base=0.1))
# print(epsilon)
# # # kernel = ["rbf", "poly"]
# param_grid = [
#     {
#         "C": C,
#         "gamma": gamma,
#         "epsilon": epsilon,
#         # "kernel": kernel,
#     },
# ]

# svr = SVR(kernel="rbf")
# grid_search = GridSearchCV(
#     svr, param_grid, cv=10, scoring="neg_mean_absolute_percentage_error"
# )
# grid_search.fit(X, y)
# print(grid_search.best_params_)
# print(grid_search.best_score_)
# results = pd.DataFrame(grid_search.cv_results_)
# print(results)
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


# # {'C': 4.900000000000002, 'epsilon': 0.010000000000000002, 'gamma': 2.8}

# svr = SVR(C=4.9, epsilon=0.01, gamma=2.8)
# score = cross_validate(
#     svr,
#     X,
#     y,
#     cv=10,
#     scoring=("r2", "neg_mean_absolute_percentage_error"),
# )
# predict = cross_val_predict(
#     svr,
#     X,
#     y,
#     cv=10,
# )
# fig, ax = plt.subplots(figsize=(6, 4.5))

# sns.scatterplot(x=y, y=predict)
# sns.lineplot(x=[0, 1], y=[0, 1])
# # ax.scatter(y, y1)
# ax.set_xlabel("Actual normalized EMY ")
# ax.set_ylabel("Predicted normalized EMY")
# plt.subplots_adjust(bottom=0.15)
# plt.subplots_adjust(left=0.165)
# plt.subplots_adjust(right=0.95)
# plt.subplots_adjust(top=0.95)
# plt.gcf().text(0.6, 0.8, "R$^2$ = %.4f" % (abs(score["test_r2"].mean())), fontsize=18)

# fig.savefig("SVR CV predicted.svg")
# print(pd.DataFrame(score))

# mean = []
# mean.append(score["test_r2"].mean())
# mean.append(score["test_neg_mean_absolute_percentage_error"].mean())
# score2 = pd.DataFrame(
#     abs(score["test_neg_mean_absolute_percentage_error"]),
#     index=np.arange(1, 11, 1),
# ).T
# print(score2)
# fig3, ax = plt.subplots(figsize=(6, 4.5))
# sns.barplot(score2)
# ax.set(xlabel="Round in CV of SVR", ylabel="MAPE in test")
# ax.set_ylim([0, 0.18])
# from matplotlib.ticker import MultipleLocator

# y_major_locator = MultipleLocator(0.02)
# ax.yaxis.set_major_locator(y_major_locator)
# plt.subplots_adjust(bottom=0.15)
# plt.subplots_adjust(left=0.165)
# plt.subplots_adjust(right=0.95)
# plt.subplots_adjust(top=0.95)
# plt.gcf().text(
#     0.4,
#     0.9,
#     "mean MAPE = %.2f%%"
#     % (100 * abs(score["test_neg_mean_absolute_percentage_error"].mean())),
#     fontsize=18,
# )

# from matplotlib import ticker

# ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

# # ax.text(x=8, y=0.03, s="%.4f\n%.4f" % (mean[0], mean[1]))
# fig3.savefig("SVR error.svg")


# svr.fit(X, y)
# y1 = svr.predict(X)
# fig, ax = plt.subplots()
# ax = sns.scatterplot(x = y, y = y1)
# # ax.scatter(y, y1)
# ax.set_xlabel("predicted normalized EMY ")
# ax.set_ylabel("actual normalized EMY")
# ax = sns.lineplot(x=[0, 1], y=[0, 1])
# fig.savefig("SVR result.svg")


# importance = svr.feature_importances_
# # summarize feature importance
# # for i, v in enumerate(importance):
# #     print("Feature: %0d, Score: %.5f" % (i, v))
# # plot feature importance
# df = pd.DataFrame(list(importance))
# df.index = Sourcedata.columns[:-1]
# df = df.sort_values(by=0, ascending=False)
# df = df.T
# fig2, ax = plt.subplots()
# ax = sns.barplot(df)
# ax.set(xlabel="Feature", ylabel="Feature importance score")
# # plt.subplots_adjust(bottom=0.1)
# # plt.subplots_adjust(left=0.1)
# # plt.subplots_adjust(right=0.95)
# # plt.subplots_adjust(top=0.95)
# fig2.savefig("feture importance.svg")

# # plt.show()

# X = data_scaled[:, :-1]
# y = data_scaled[:, -1]
# # X_test = data_scaled[-50:, :-1]

# # y_test = data_scaled[-50:, -1]
# model = grid_search.best_estimator_
# # fit the model
# model.fit(X, y)
# y1 = model.predict(X)
# fig, ax = plt.subplots()
# ax.scatter(y1, y)
# # MSEarray = metrics.mean_squared_error(y1, y_test)
# # print(MSEarray)
# # get importance


# importance = model.feature_importances_
# # summarize feature importance
# # for i, v in enumerate(importance):
# #     print("Feature: %0d, Score: %.5f" % (i, v))
# # plot feature importance
# df = pd.DataFrame(data=[list(Sourcedata.columns)[:-1], list(importance)]).T
# df = df.sort_values(by=1, ascending=False)
# print(df)
# fig2, ax = plt.subplots()
# ax.bar(
#     df[0],
#     df[1],
#     tick_label=df[0],
# )
# fig2.autofmt_xdate(rotation=45)
# plt.show()
