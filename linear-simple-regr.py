import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('fuelcons.csv')
df.columns
x=df.ENGINESIZE
y=df.CO2EMISSIONS
plt.scatter(x,y, color='blue')
plt.xlabel('Engine size')
plt.ylabel('CO2 emission')

plt.savefig('emission-vs-enginesize.png', dpi=300)
#plt.show()

#generate train and test data

msk=np.random.rand(len(df)) <0.8
train=df[msk]
test=df[~msk]
train_x=np.asanyarray(train[['ENGINESIZE']])
test_x=np.asanyarray(test[['ENGINESIZE']])
train_y=np.asanyarray(train[['CO2EMISSIONS']])
test_y=np.asanyarray(test[['CO2EMISSIONS']])

#model train data
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(train_x,train_y)
print('coef:',regr.coef_)
print('intercept:',regr.intercept_)

plt.scatter(train_x,train_y, color='blue')
plt.plot(train_x,train_x*39.10553097+125.1895725, color='red')
plt.xlabel('Engine size')
plt.ylabel('CO2 emission')
plt.savefig('data-and-model.png', dpi=300)
#plt.show()

#evaluation

#mean square error

np.mean((regr.predict(test_x)-test_y)**2)

from sklearn.metrics import r2_score
r2_score(test_y,regr.predict(test_x))



