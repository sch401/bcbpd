from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
from matplotlib.ticker import MultipleLocator
from matplotlib import pyplot as plt
from sklearn.model_selection import (
    cross_validate,
    cross_val_predict,
)
import numpy as np
import seaborn as sns

plt.rcParams["font.size"] = 18
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
Sourcedata = pd.read_excel("dataset_origin.xlsx", sheet_name="Sheet1")
standardScaler = MaxAbsScaler()
standardScaler.fit(Sourcedata)
data_scaled = standardScaler.transform(Sourcedata)

X = data_scaled[:, :-1]
y = data_scaled[:, -1]


# ScoreAll = []
# for i in range(175, 200,1):
#     RF = RandomForestRegressor(n_estimators=i, random_state=66)
#     score = cross_val_score(RF, X, y, cv=10,scoring="neg_mean_absolute_percentage_error").mean()
#     ScoreAll.append([i, score])
# ScoreAll = np.array(ScoreAll)
# plt.figure()
# plt.plot(ScoreAll[:, 0], ScoreAll[:, 1])
# plt.show()
# n_estimators = 182

# ScoreAll = []
# for i in range(10,20):
#     RF = RandomForestRegressor(n_estimators=182, max_depth = i,random_state = 66)
#     score = cross_val_score(RF, X, y, cv=10,scoring="neg_mean_absolute_percentage_error").mean()
#     ScoreAll.append([i,score])
# ScoreAll = np.array(ScoreAll)
# plt.figure()
# plt.plot(ScoreAll[:, 0], ScoreAll[:, 1])
# plt.show()
#  max_depth = 11

# ScoreAll = []
# for i in range(2, 10, 1):
#     RF = RandomForestRegressor(
#         n_estimators=182, max_depth=11, min_samples_leaf=i, random_state=66
#     )
#     score = cross_val_score(RF, X, y, cv=10, scoring="neg_mean_absolute_percentage_error").mean()
#     ScoreAll.append([i, score])
# ScoreAll = np.array(ScoreAll)
# plt.figure()
# plt.plot(ScoreAll[:, 0], ScoreAll[:, 1])
# plt.show()
# min_samples_leaf = 2


# max_depth = list(np.arange(4, 8, 2))
# n_estimators = list(np.arange(185, 195, 2))
# min_samples_leaf = [2, 3, 4, 5, 6, 7, 8]
# param_grid = [
#     {
#         "max_depth": max_depth,
#         "n_estimators": n_estimators,
#         "min_samples_leaf": min_samples_leaf,
#     },
# ]

# forest_reg = RandomForestRegressor(random_state=66)
# grid_search = GridSearchCV(
#     forest_reg, param_grid, cv=10, scoring="neg_mean_absolute_percentage_error"
# )
# grid_search.fit(X, y)
# print(grid_search.best_params_)
# print(grid_search.best_score_)

# {'max_depth': 6, 'min_samples_leaf': 5, 'n_estimators': 187}


RF = RandomForestRegressor(
    n_estimators=187, max_depth=6, min_samples_leaf=5, random_state=66
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

fig, ax = plt.subplots(figsize=(6, 4.5))

fig = sns.scatterplot(x=y, y=predict)
ax.set_xlabel("Actual normalized EMY ")
ax.set_ylabel("Predicted normalized EMY")
fig = sns.lineplot(x=[0, 1], y=[0, 1])
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.165)
plt.subplots_adjust(right=0.95)
plt.subplots_adjust(top=0.95)
plt.gcf().text(0.6, 0.8, "R$^2$ = %.4f" % (abs(score["test_r2"].mean())), fontsize=18)
plt.savefig("RF_maxabsscaler_mean.svg")
from matplotlib import ticker

print(pd.DataFrame(score))

mean = []
mean.append(score["test_r2"].mean())
mean.append(score["test_neg_mean_absolute_percentage_error"].mean())
score2 = pd.DataFrame(
    abs(score["test_neg_mean_absolute_percentage_error"]), index=np.arange(1, 11, 1)
).T
pd.DataFrame(mean).to_csv("RF_maxabsscaler_mean.csv")

fig3, ax = plt.subplots(figsize=(6, 4.5))
fig3 = sns.barplot(score2)
ax.set(xlabel="Round in CV of RF", ylabel="MAPE in test")
plt.gcf().text(
    0.4,
    0.9,
    "mean MAPE = %.2f%%"
    % (100 * abs(score["test_neg_mean_absolute_percentage_error"].mean())),
    fontsize=18,
)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ymajor = MultipleLocator(0.1)
ax.yaxis.set_major_locator(ymajor)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.165)
plt.subplots_adjust(right=0.95)
plt.subplots_adjust(top=0.95)

plt.savefig("RF error of maxabsscaler.svg")
