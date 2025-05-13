from sklearn.linear_model import LinearRegression
import numpy as np, matplotlib.pyplot as plt
def loc_reg(x,X,y,tau):
    weights=np.exp(-(x-X)**2/(2*tau**2))
    model=LinearRegression()
    model.fit(X.reshape(-1,1),y,weights)
    return model.predict([[x]])
x_train=np.linspace(1,5,100)
y_train=y = np.sin(x_train) + 0.1 * np.random.randn(100)
x_test=np.linspace(1,5,100)
y_val=[loc_reg(xi,x_train,y_train,0.3) for xi in x_test]
plt.scatter(x_train,y_train,c='r')
plt.plot(x_test,y_val,c='b')