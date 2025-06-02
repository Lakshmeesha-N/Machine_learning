import seaborn as sns ,pandas as pd
from sklearn.datasets import load_iris;from sklearn.decomposition import PCA
pca=PCA(2)
data=pd.DataFrame(pca.fit_transform(load_iris().data),columns=['pc1','pc2'])
data['label']=load_iris().target
sns.scatterplot(x=data['pc1'],y=data['pc2'],hue=load_iris().target , legend=False)
plt.title('PCA on Iris Dataset')
