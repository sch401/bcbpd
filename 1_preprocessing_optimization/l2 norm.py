from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import normalize
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import (
    cross_validate,
    cross_val_predict,
)
import numpy as np
import seaborn as sns
from matplotlib import ticker

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
Sourcedata = pd.read_excel("副本MLAD_SourceData (3) - 副本.xlsx", sheet_name="Sheet1")
data_scaled = normalize(Sourcedata)

X = data_scaled[:, :-1]
y = data_scaled[:, -1]


# ScoreAll = []
# for i in range(20, 100, 1):
#     RF = RandomForestRegressor(n_estimators=i, random_state=66)
#     score = cross_val_score(RF, X, y, cv=10,scoring="neg_mean_absolute_percentage_error").mean()
#     ScoreAll.append([i, score])
# ScoreAll = np.array(ScoreAll)
# plt.figure()
# plt.plot(ScoreAll[:, 0], ScoreAll[:, 1])
# plt.show()
# n_estimators = 74

# ScoreAll = []
# for i in range(5, 20):
#     RF = RandomForestRegressor(n_estimators=74, max_depth=i, random_state=66)
#     score = cross_val_score(
#         RF, X, y, cv=10, scoring="neg_mean_absolute_percentage_error"
#     ).mean()
#     ScoreAll.append([i, score])
# ScoreAll = np.array(ScoreAll)
# plt.figure()
# plt.plot(ScoreAll[:, 0], ScoreAll[:, 1])
# plt.show()
#  max_depth = 10

# ScoreAll = []
# for i in range(2, 40, 2):
#     RF = RandomForestRegressor(
#         n_estimators=74, max_depth=10, min_samples_leaf=i, random_state=66
#     )
#     score = cross_val_score(RF, X, y, cv=10, scoring="neg_mean_absolute_percentage_error").mean()
#     ScoreAll.append([i, score])
# ScoreAll = np.array(ScoreAll)
# plt.figure()
# plt.plot(ScoreAll[:, 0], ScoreAll[:, 1])
# plt.show()
# min_samples_leaf = 2


# max_depth = list(np.arange(8, 12, 1))
# n_estimators = list(np.arange(72, 76, 1))
# param_grid = [
#     {
#         "max_depth": max_depth,
#         "n_estimators": n_estimators,

#     },
# ]

# forest_reg = RandomForestRegressor(random_state=66)
# grid_search = GridSearchCV(
#     forest_reg, param_grid, cv=10, scoring="neg_mean_absolute_percentage_error"
# )
# grid_search.fit(X, y)
# print(grid_search.best_params_)
# print(grid_search.best_score_)

# {'max_depth': 11, 'n_estimators': 75}

RF = RandomForestRegressor(
    n_estimators=75,
    max_depth=11,
    random_state=66,
)
score = cross_validate(
    RF,
    X,
    y,
    cv=10,
    scoring=("r2", "neg_mean_absolute_percentage_error"),
)
predict = cross_val_predict(
    RF,
    X,
    y,
    cv=10,
)


print(pd.DataFrame(score))


mean = []
mean.append(score["test_neg_mean_absolute_percentage_error"].mean())
mean.append(score["test_neg_mean_absolute_percentage_error"].mean())
score2 = pd.DataFrame(
    abs(score["test_neg_mean_absolute_percentage_error"]), index=np.arange(1, 11, 1)
).T
plt.rcParams["font.size"] = 18
fig3, ax = plt.subplots(figsize=(6, 4.5))
fig3 = sns.barplot(score2)
plt.gcf().text(
    0.4,
    0.9,
    "mean MAPE = %.2f%%"
    % (100 * abs(score["test_neg_mean_absolute_percentage_error"].mean())),
    fontsize=18,
)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax.set(xlabel="Round in CV of RF", ylabel="MAPE in test")
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.165)
plt.subplots_adjust(right=0.95)
plt.subplots_adjust(top=0.95)
plt.savefig("RF error.svg")

fig, ax = plt.subplots(figsize=(6, 4.5))
fig = sns.scatterplot(x=y, y=predict)
ax.set_xlabel("Actual normalized EMY")
ax.set_ylabel("Predicted normalized EMY")
plt.gcf().text(0.6, 0.8, "R$^2$ = %.4f" % (abs(score["test_r2"].mean())), fontsize=18)
ax.plot([0, 1], [0, 1])
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.165)
plt.subplots_adjust(right=0.95)
plt.subplots_adjust(top=0.95)
plt.savefig("RF result.svg")
RF.fit(X, y)
importance = RF.feature_importances_
# summarize feature importance
# for i, v in enumerate(importance):
#     print("Feature: %0d, Score: %.5f" % (i, v))
# plot feature importance
df = pd.DataFrame(list(importance))
df.index = Sourcedata.columns[:-1]
df = df.sort_values(by=0, ascending=False)
df = df.T

fig2, ax = plt.subplots(figsize=(6, 12))
fig2 = sns.barplot(df, orient="h")
plt.xscale("log")
# plt.subplots_adjust(bottom=0.13)
plt.subplots_adjust(left=0.3)
ax.set_xlabel(
    "Feature importance score",
    fontsize=26,
)
ax.set_ylabel(
    "Feature",
    fontsize=26,
)
# plt.subplots_adjust(right=0.95)
# plt.subplots_adjust(top=0.95)
# plt.gcf().text(
#     0.6,
#     0.8,
#     "mean MAPE = %.2f%%"
#     % (100 * abs(score["test_neg_mean_absolute_percentage_error"].mean())),
#     fontsize=12,
# )

plt.savefig("feature importance.svg")
df.to_csv("importance score.csv")


# plt.show()

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

# RF = RandomForestRegressor(random_state=66)
# score = cross_validate(
#     RF,
#     X,
#     y,
#     cv=10,
#     scoring=("neg_mean_absolute_percentage_error", "neg_mean_absolute_percentage_error"),
# )

# print(pd.DataFrame(score))

# mean = []
# mean.append(score["test_neg_mean_absolute_percentage_error"].mean())
# mean.append(score["test_neg_mean_absolute_percentage_error"].mean())
# score = pd.DataFrame(
#     abs(score["test_neg_mean_absolute_percentage_error"]), index=np.arange(1, 11, 1)
# ).T

# fig3, ax = plt.subplots()
# ax = sns.barplot(score)
# ax.set(xlabel="Round in CV of RF", ylabel="MSE")
# ax.text(x=8, y=0.03, s="%.4f\n%.4f" % (mean[0], mean[1]))
# fig3.savefig("RF error normal with y.svg")
# RF.fit(X, y)
# y1 = RF.predict(X)
# fig, ax = plt.subplots()
# ax.scatter(y, y1)
# ax.set_xlabel("EMY tested")
# ax.set_ylabel("EMY predicted")
# ax.plot([0, 1], [0, 1])
# fig.savefig("RF result normal with y.svg")
