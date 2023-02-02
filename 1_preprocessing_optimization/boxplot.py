import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 13
# sns.set_theme(
#     # style="ticks",
#     # # font_scale=1.5,
#     # color_codes=True,
#     # rc=None,
# )
# df = sns.load_dataset("titanic")
# print(df)
Sourcedata = pd.read_excel("dataset_original.xlsx", sheet_name="Sheet1")
print(Sourcedata)
Sourcedata["N\n(TS%)"] = 10 * Sourcedata["N\n(TS%)"]

Sourcedata["TMY\n(mL/gVS)"] = 0.1 * Sourcedata["TMY\n(mL/gVS)"]
Sourcedata["H\n(TS%)"] = 10 * Sourcedata["H\n(TS%)"]

Sourcedata["EMY\n(mL/gVS)"] = 0.1 * Sourcedata["EMY\n(mL/gVS)"]
Sourcedata.rename(
    columns={
        "EMY\n(mL/gVS)": "0.1EMY\n(mL/gVS)",
        "N\n(TS%)": "10N\n(TS%)",
        "H\n(TS%)": "10H\n(TS%)",
        "TMY\n(mL/gVS)": "0.1TMY\n(mL/gVS)",
    },
    inplace=True,
)
fig, ax = plt.subplots(figsize=(6, 10))
fig = sns.boxplot(Sourcedata, orient="h")
ax.set(xlabel="Value", ylabel="Feature")
ax.set_xlabel("Value", fontsize=22)
ax.set_ylabel(
    "Feature",
    fontsize=22,
)
ax.tick_params(labelsize=16)
plt.subplots_adjust(left=0.3)
plt.subplots_adjust(top=0.9)
plt.subplots_adjust(bottom=0.1)
plt.savefig("boxplot.svg")
info = pd.concat(
    [
        pd.DataFrame([Sourcedata.mean(), Sourcedata.min(), Sourcedata.max()]),
        Sourcedata.quantile([0.25, 0.5, 0.75]),
    ]
)
info.to_csv("info.csv")
