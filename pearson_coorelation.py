import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.5)
plt.rcParams["font.family"] = "Times New Roman"

Sourcedata = pd.read_excel("dataset_original.xlsx", sheet_name="Sheet1")
data_scaled = normalize(Sourcedata)

dcorr_norm = pd.DataFrame(data_scaled).corr(method="pearson")
f, ax = plt.subplots(figsize=(12, 10))
f = sns.heatmap(
    data=dcorr_norm,
    annot=True,  
    fmt=".2f", 
    xticklabels=Sourcedata.columns,
    yticklabels=Sourcedata.columns,
    annot_kws={"size": 13},
)

plt.subplots_adjust(bottom=0.1)
plt.subplots_adjust(left=0.1)
plt.subplots_adjust(right=1.05)
plt.subplots_adjust(top=0.9)
plt.savefig("pearson_normalized.svg")


dcorr = pd.DataFrame(Sourcedata).corr(method="pearson")
d, ax = plt.subplots(figsize=(12, 10))
d = sns.heatmap(
    data=dcorr,
    annot=True, 
    fmt=".2f", 
    xticklabels=Sourcedata.columns,
    yticklabels=Sourcedata.columns,
    annot_kws={"size": 13},
)


plt.subplots_adjust(bottom=0.1)
plt.subplots_adjust(left=0.1)
plt.subplots_adjust(right=1.05)
plt.subplots_adjust(top=0.9)
plt.savefig("pearson.svg")
