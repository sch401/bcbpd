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
plt.rcParams["font.size"] = 18
Sourcedata = pd.read_excel("dataset_origin.xlsx", sheet_name="Sheet2")
data_scaled = normalize(Sourcedata)

X = data_scaled[:, :-1]
y = data_scaled[:, -1]


svr = SVR(C=4.9, epsilon=2.154e-3, gamma=1.8)
score = cross_validate(
    svr,
    X,
    y,
    cv=10,
    scoring=("r2", "neg_mean_absolute_percentage_error"),
)
predict = cross_val_predict(
    svr,
    X,
    y,
    cv=10,
)
fig, ax = plt.subplots(figsize=(6, 4.5))

sns.scatterplot(x=y, y=predict)
sns.lineplot(x=[0, 1], y=[0, 1])
# ax.scatter(y, y1)
ax.set_xlabel("Actual normalized EMY ")
ax.set_ylabel("Predicted normalized EMY")
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.165)
plt.subplots_adjust(right=0.95)
plt.subplots_adjust(top=0.95)
plt.gcf().text(0.6, 0.8, "R$^2$ = %.4f" % (abs(score["test_r2"].mean())), fontsize=18)

fig.savefig("svmoptimize/SVR CV predicted.svg")
print(pd.DataFrame(score))

mean = []
mean.append(score["test_r2"].mean())
mean.append(score["test_neg_mean_absolute_percentage_error"].mean())
score2 = pd.DataFrame(
    abs(score["test_neg_mean_absolute_percentage_error"]),
    index=np.arange(1, 11, 1),
).T
print(score2)
fig3, ax = plt.subplots(figsize=(6, 4.5))
sns.barplot(score2)
ax.set(xlabel="Round in CV of SVR", ylabel="MAPE in test")
ax.set_ylim([0, 0.18])
from matplotlib.ticker import MultipleLocator

y_major_locator = MultipleLocator(0.02)
ax.yaxis.set_major_locator(y_major_locator)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.165)
plt.subplots_adjust(right=0.95)
plt.subplots_adjust(top=0.95)
plt.gcf().text(
    0.4,
    0.9,
    "mean MAPE = %.2f%%"
    % (100 * abs(score["test_neg_mean_absolute_percentage_error"].mean())),
    fontsize=18,
)

from matplotlib import ticker

ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

# ax.text(x=8, y=0.03, s="%.4f\n%.4f" % (mean[0], mean[1]))
fig3.savefig("svmoptimize/SVR error.svg")


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
