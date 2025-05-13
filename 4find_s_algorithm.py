import pandas as pd
data = pd.read_csv(r"C:\Users\laksh\Downloads\data.csv")
hyp = ['*'] * (len(data.columns) - 1)
for _, row in data.iterrows():
    if row.iloc[-1] in ['True', 'yes']: 
        for i in range(len(data.columns) - 1):
            if hyp[i] == '*' or hyp[i] == row.iloc[i]:
                hyp[i] = row.iloc[i] 
            else:
                hyp[i] = '?'
print(hyp)