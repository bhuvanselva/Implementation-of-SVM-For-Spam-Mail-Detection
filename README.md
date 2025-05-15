# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2.Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: BHUVANESHWARI S
RegisterNumber:  212222220008
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
# Result Output:
![image](https://github.com/user-attachments/assets/5c1d6a61-1689-4553-8bf3-6a4bf49d4902)

# data.head()
![image](https://github.com/user-attachments/assets/4811debd-e23e-4a42-bd9e-7a2d1f8223ac)

# ata.info():

![image](https://github.com/user-attachments/assets/835a57c3-0d36-4b41-bd94-105fb165b3ab)

# data.isnull().sum()
![image](https://github.com/user-attachments/assets/038e5b3d-c0cf-43f0-9324-a4c465d737b3)

![image](https://github.com/user-attachments/assets/d55f8363-a31c-421b-9c85-b65ecebe077b)

# Y_prediction Value

![image](https://github.com/user-attachments/assets/146c7aa5-440b-48c9-8fa5-8a2aaa44828d)

# Accuracy Value:
![image](https://github.com/user-attachments/assets/d9b20f21-81ad-433a-8355-b6e9438a8734)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
