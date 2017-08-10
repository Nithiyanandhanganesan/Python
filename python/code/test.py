# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:29:57 2017

@author: 753914
"""
import pandas as pd
import seaborn as sns
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import numpy as np

df_train = pd.read_csv('C:/Users/753914/Desktop/my_technical/machine_learning/practical_machine_learning/linear_regression/code/house_price_pred/train.csv')
df_test = pd.read_csv('C:/Users/753914/Desktop/my_technical/machine_learning/practical_machine_learning/linear_regression/code/house_price_pred/test.csv')
#print(df_train.columns)

#print(df_train['SalePrice'].describe())
#sns.distplot(df_train['SalePrice']);
#print("Skewness: %f" % df_train['SalePrice'].skew())
#print("Kurtosis: %f" % df_train['SalePrice'].kurt())

x=df_train.loc[:]['GrLivArea']
y=df_train.loc[:]['SalePrice']

length=df_train['GrLivArea'].count()
x = x.values.reshape(length, 1)
y = y.values.reshape(length, 1)

regr = linear_model.LinearRegression()
regr.fit(x, y)


x_test=df_test.loc[:]['GrLivArea']
length_test=df_test['GrLivArea'].count()
x_test = x_test.values.reshape(length_test, 1)

print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % np.mean((regr.predict(x) - x) ** 2))


plt.scatter(x, y,  color='black')
plt.plot(x, regr.predict(x), color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

#output = pd.DataFrame(data=np.array([x,regr.predict(x)]))
test=pd.DataFrame(x_test)
#`for index, row in test.iterrows() :    
#     print(row[0],regr.predict(row[0]))







    
          


