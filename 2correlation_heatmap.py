import pandas as pd ,seaborn as sns
from sklearn.datasets import fetch_california_housing
df=fetch_california_housing(as_frame=True).frame
sns.heatmap(df.corr());sns.pairplot(df)