import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import fetch_california_housing
df = fetch_california_housing(as_frame=True).frame 
plt.figure(figsize=(10,15))
for i, feature in enumerate(df.columns):    
    q1, q3 = df[feature].quantile(0.25), df[feature].quantile(0.75)
    lb, up = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    print(f'{feature},{len(df[(df[feature] < lb) | (df[feature] > up)])}')
    plt.subplot(6, 3, i+1)
    sns.histplot(df[feature], color='g')
    plt.subplot(6, 3, i+10)
    sns.boxplot(x=df[feature], color='r')
plt.tight_layout()
print(df.describe()) 
