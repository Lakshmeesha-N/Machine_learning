import numpy as np,seaborn as sns
from collections import Counter 
def knn(x,xi,y,k):
    dis=list(zip(abs((x)-( np.array(xi))),y))
    val = sorted(dis, key=lambda x: x[0])[:k]
    return Counter(i for _ , i in val ).most_common(1)[0][0]
data=np.random.rand(100)
l=['c1' if i > 0.5 else 'c2' for i in data]
val=[ knn(data[:50],xi,l[:50],3) for xi in data[50:]]
sns.scatterplot(x=data[:50],y=[1]*50,hue=l[0:50],palette=['r','g'],marker='*' )
sns.scatterplot(x=data[50:], y=[0.5]*50, hue=val,palette=['g','r'])
print("Point|Predct| Actual Value")# last 3 lines optional 
for i in range(50):
    print(f"{data[50+i]:.3f}| {val[i]}   |{l[50+i]}")
