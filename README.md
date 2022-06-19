# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Start the program
2. import pandas as pd and read required csv file
3. print data.head(),data.info()
4. assign x=data["v1"].values,y=data["v2"].values
5. import train_test_split and split the data into test and test datasets as test_size=0.2, random_state=0
6. import CountVectorizer and create object for CountVectorizer()
7. import SVC and print y_pred
8. import metrics and print accuracy
9. Stop the program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: S Saichandran
RegisterNumber:  212220040138
*/
import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='latin-1')
data.head()
data.info()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer #CountVectorizer is a method to convert text into numeric data. The text is transformed to a sparse matrix
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
![SVM For Spam Mail Detection](/1.data.head().PNG)
![SVM For Spam Mail Detection](/2.data.info().PNG)
![SVM For Spam Mail Detection](/3.y_pred.PNG)
![SVM For Spam Mail Detection](/4.accuracy().PNG)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
