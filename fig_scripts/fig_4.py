import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("outputs/robin_dock_raw.csv")

#sns.displot(data=df, col='pocket_id', x='raw_score', hue='is_active')
sns.violinplot(data=df, x="pocket_id", y="raw_score", hue="is_active")
plt.show()
