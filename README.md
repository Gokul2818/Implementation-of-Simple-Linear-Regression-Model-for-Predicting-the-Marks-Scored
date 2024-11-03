# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: GOKUL S
RegisterNumber:  24004336
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE =',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE =',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![S1](https://github.com/user-attachments/assets/a7b1f2d3-4789-4336-9493-1873ea420c32)
![S2](https://github.com/user-attachments/assets/bbef3f2c-eaab-48d4-b78e-41671a64227f)
![S3](https://github.com/user-attachments/assets/02e0034d-7a76-46f9-8d31-a8e2117c9538)
![S4](https://github.com/user-attachments/assets/32e95a66-cde1-412f-8e4b-7d8b40c070a3)
![S5](https://github.com/user-attachments/assets/491be272-83d8-48ad-8bd5-6abd439c116c)
![S6](https://github.com/user-attachments/assets/9a61b0ea-0273-46b2-b0f6-7371098d0a71)
![S7](https://github.com/user-attachments/assets/72c163dd-c2bc-4139-bc9a-fecaae59eda5)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
