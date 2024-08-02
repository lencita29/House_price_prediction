

import pandas as pd
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt

#Load Dataset from Local Directory

#from google.colab import files
#uploaded = files.upload()

#Load Dataset

data2 = pd.read_csv('data2.csv')

#Load Summarize

print(data2.shape)
print(data2.head(5))

#Visualize Dataset

plt.xlabel('Level')
plt.ylabel('Salary')
plt.scatter(data2.Level,data2.Salary,color='red',marker='*')
plt.show()

#Segregate Dataset into Input X & Output Y

X = data2.drop('Salary',axis='columns')
X

Y = data2.Salary
Y

#Training Dataset using Linear Regression

model = LinearRegression()
model.fit(X,Y)

#Predicted Price for Land sq.Feet of custom values

x=40000
LandAreainSqFt=[[x]]
PredictedmodelResult = model.predict(LandAreainSqFt)
print(PredictedmodelResult)


m=model.coef_
print(m)

#Intercept - b

b=model.intercept_
print(b)# Y=mx+b
#x is Independant variable - Input - area

y = m*x + b
print("The Price of {0} Square feet Land is: {1}".format(x,y[0]))
