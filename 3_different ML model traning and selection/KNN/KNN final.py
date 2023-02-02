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

# from sklearn.svm import KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor

Sourcedata = pd.read_excel("副本MLAD_SourceData (3) - 副本.xlsx", sheet_name="Sheet2")
data_scaled = normalize(Sourcedata)
X = data_scaled[:, :-1]
y = data_scaled[:, -1]


# ScoreAll = []
# for i in range(1, 10, 1):
#     knn = KNeighborsRegressor(n_neighbors=i)
#     score = cross_val_score(knn, X, y, cv=10,scoring="neg_mean_absolute_percentage_error").mean()
#     ScoreAll.append([i, score])
# ScoreAll = np.array(ScoreAll)
# plt.figure()
# plt.plot(ScoreAll[:, 0], ScoreAll[:, 1])
# plt.show()
# # # n_neighbors = 4

# ScoreAll = []
# for i in range(2, 10, 1):
#     knn = KNeighborsRegressor(n_neighbors=4, p=i)
#     score = cross_val_score(
#         knn, X, y, cv=10, scoring="neg_mean_absolute_percentage_error"
#     ).mean()
#     ScoreAll.append([i, score])
# ScoreAll = np.array(ScoreAll)
# plt.figure()
# plt.plot(ScoreAll[:, 0], ScoreAll[:, 1])
# plt.show()
# p = 2


# n_neighbors = list(np.arange(2, 6, 1))
# p = np.array([x * 0.1 for x in range(10,50,5)])
# p = list(np.around(p, 3))
# param_grid = [
#     {
#         "n_neighbors": n_neighbors,
#         "p": p,

#     },
# ]
# from sklearn.model_selection import GridSearchCV
# knn = KNeighborsRegressor()
# grid_search = GridSearchCV(
#     knn, param_grid, cv=10, scoring="neg_mean_absolute_percentage_error"
# )
# grid_search.fit(X, y)
# print(grid_search.best_params_)
# print(grid_search.best_score_)
# results = pd.DataFrame(grid_search.cv_results_)
# results.to_csv("KNNoptimize\knn遍历neighbourp.csv")






# # {'n_neighbors': 4, 'p': 2, 'weights': 'distance'}

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
knn = KNeighborsRegressor(n_neighbors=4, p=2)
score = cross_validate(
    knn,
    X,
    y,
    cv=10,
    scoring=("r2", "neg_mean_absolute_percentage_error"),
)
predict = cross_val_predict(
    knn,
    X,
    y,
    cv=10,
)
print(pd.DataFrame(score))
fig, ax = plt.subplots(figsize=(6, 4.5))
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.165)
plt.subplots_adjust(right=0.95)
plt.subplots_adjust(top=0.95)
sns.scatterplot(x=y, y=predict)
# ax.scatter(y, y1)
ax.set_xlabel("Actual normalized EMY ")
ax.set_ylabel("Predicted normalized EMY")
ax = sns.lineplot(x=[0, 1], y=[0, 1])
plt.gcf().text(0.6, 0.8, "R$^2$ = %.4f" % (abs(score["test_r2"].mean())), fontsize=18)

fig.savefig("KNeighborsRegressor CV predicted.svg")

mean = []
mean.append(score["test_neg_mean_absolute_percentage_error"].mean())
mean.append(score["test_neg_mean_absolute_percentage_error"].mean())
score2 = pd.DataFrame(
    abs(score["test_neg_mean_absolute_percentage_error"]),
    index=np.arange(1, 11, 1),
).T

fig3, ax = plt.subplots(figsize=(6, 4.5))
plt.gcf().text(
    0.4,
    0.9,
    "mean MAPE = %.2f%%"
    % (100 * abs(score["test_neg_mean_absolute_percentage_error"].mean())),
    fontsize=18,
)
sns.barplot(score2)

ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax.set_ylim([0, 0.18])
from matplotlib.pyplot import MultipleLocator

y_major_locator = MultipleLocator(0.02)
ax.yaxis.set_major_locator(y_major_locator)
ax.set(xlabel="Round in CV of KNN", ylabel="MAPE in test")
# ax.text(x=8, y=0.03, s="%.4f\n%.4f" % (mean[0], mean[1]))
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.165)
plt.subplots_adjust(right=0.95)
plt.subplots_adjust(top=0.95)

fig3.savefig("KNeighborsRegressor error.svg")
