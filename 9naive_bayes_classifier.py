import matplotlib.pyplot as plt,numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split ,cross_val_score
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.naive_bayes import GaussianNB
X, y = fetch_olivetti_faces(shuffle=True, random_state=42, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
gnb = GaussianNB().fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred, zero_division=1))
cross_val_accuracy = cross_val_score(gnb, X, y, cv=5, scoring='accuracy')
print(f'\nCross-validation accuracy: {cross_val_accuracy.mean() * 100:.2f}%')
plt.figure(figsize=(12, 8))
for i in range(15):
    plt.subplot(3, 5, i + 1)  
    plt.imshow(X_test[i].reshape(64, 64), cmap="gray") 
    plt.title(f"True: {y_test[i]} | Predicted: {y_pred[i]}", color="red", fontsize=10)
