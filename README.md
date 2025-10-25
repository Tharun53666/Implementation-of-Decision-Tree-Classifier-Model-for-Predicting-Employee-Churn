# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import pandas
Import Decision tree classifier
Fit the data in the model
Find the accuracy score 

## Program:
```
      import pandas as pd
      data=pd.read_csv("Employee.csv")
      print("data.head():")
      data.head()

      
      print("data.info():")
      data.info()

      print("isnull() and sum():")
      data.isnull().sum()

      print("data value counts():")
      data["left"].value_counts()

      from sklearn.preprocessing import LabelEncoder
      le=LabelEncoder()

      print("data.head() for Salary:")
      data["salary"]=le.fit_transform(data["salary"])
      data.head()
      print("x.head():")
      x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
      x.head()

      y=data["left"]
      from sklearn.model_selection import train_test_split
      x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
      from sklearn.tree import DecisionTreeClassifier
      dt=DecisionTreeClassifier(criterion="entropy")
      dt.fit(x_train,y_train)
      y_pred=dt.predict(x_test)

      print("Accuracy value:")
      from sklearn import metrics
      accuracy=metrics.accuracy_score(y_test,y_pred)
      accuracy

      print("Data Prediction:")
      dt.predict([[0.5,0.8,9,260,6,0,1,2]])

      from sklearn.tree import plot_tree
      import matplotlib.pyplot as plt
      
      plt.figure(figsize=(8,6))
      plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
      plt.show()
```


## Output:

<img width="1232" height="228" alt="390815135-0a4fa5db-4d79-4ebd-9926-f4d1ee83d1e3" src="https://github.com/user-attachments/assets/e517c1b0-a79b-4f6e-9a3f-e4f7a6a59dd1" />


<img width="980" height="366" alt="390815192-ea8d6d09-3cc4-4875-ade1-46c9aa59e5c0" src="https://github.com/user-attachments/assets/a7701698-ab9a-4b34-b13c-3a873f677eaa" />



<img width="843" height="257" alt="390815252-9b850bb3-7c7e-4775-b2e3-789789375431" src="https://github.com/user-attachments/assets/a1b0f8c4-e105-4b9b-9cb3-228eff6caeca" />



<img width="501" height="120" alt="390815439-7b31af4d-1ce9-4ff7-b45c-938ce231cb83" src="https://github.com/user-attachments/assets/e162577b-4975-40af-8c2d-5199a6ee5abe" />


<img width="1247" height="223" alt="390815519-71654c7d-2c86-4613-8d03-a010cb8a2b11" src="https://github.com/user-attachments/assets/3384a6a4-14d8-4d8b-8067-c2e3ca149a88" />


<img width="1226" height="222" alt="390815584-5442db03-d63b-404e-ab1f-d9552a1a6a83" src="https://github.com/user-attachments/assets/b3ec8b3d-d1b3-48af-ba6e-793dcdf9b3f3" />


<img width="452" height="65" alt="390815637-727ed6aa-319b-4a29-8147-e094729d5549" src="https://github.com/user-attachments/assets/0321f40b-7963-4f0b-a308-9e8abedff731" />

<img width="1247" height="133" alt="390815954-0e78318f-49be-400e-8e9d-3d4360dc9cc3" src="https://github.com/user-attachments/assets/56b869da-c49c-4f1e-bfe7-3131806de20a" />



<img width="651" height="482" alt="390815766-4d27bab8-fef9-46a9-869c-23296ad3caac" src="https://github.com/user-attachments/assets/92dbaf6b-8d00-4125-9355-43f13464a9a3" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
