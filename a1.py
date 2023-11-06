import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier,BaggingClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv("https://raw.githubusercontent.com/Premalatha-success/Datasets/main/TaxiFare.csv")


print(data.sample(10))
print("\n")
print(data.dtypes)
print("\n")
print(data.shape)
print("\n")
print(data.describe)
print('\n')
print(data.info())

print(data['amount'].value_counts())

data['amount']=le.fit_transform(data['amount'])
data['unique_id']=le.fit_transform(data['unique_id'])
data['date_time_of_pickup']=le.fit_transform(data['date_time_of_pickup'])
data['longitude_of_pickup']=le.fit_transform(data['longitude_of_pickup'])
data['latitude_of_pickup']=le.fit_transform(data['latitude_of_pickup'])
data['longitude_of_dropoff']=le.fit_transform(data['longitude_of_dropoff'])
data['latitude_of_dropoff']=le.fit_transform(data['latitude_of_dropoff'])
data['no_of_passenger']=le.fit_transform(data['no_of_passenger'])

sns.countplot(x="unique_id",hue="amount",data=data)
plt.show()

sns.countplot(x="no_of_passengers",hue="amount",data=data)
plt.show()


correlation_mat=data.corr()
sns.heatmap(correlation_mat,annot=True,linewidths=5,cmap="YlGnBu")
plt.show()


plt.figure(figsize=(100,60))
sns.heatmap(data.isnull(),yticklabels=False)
plt.show()


X=data.drop(['amount','unique_id'],axis=1)
y=data['amount']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


abc1=AdaBoostClassifier(n_estimators=120,random_state=0)
print(abc1.fit(X_train,y_train))
print(abc1.score(X_test,y_test))

gbc1=GradientBoostingClassifier(n_estimators=120,random_state=0)
print(gbc1.fit(X_train,y_train))
print(gbc1.score(X_test,y_test))

model=LogisticRegression(solver='liblinear')
print(model.fit(X_train,y_train))
print(model.score(X_train,y_train))
print(model.score(X_test,y_test))

dtree=DecisionTreeClassifier(criterion='gini')
dtree.fit(X_train,y_train)
print(dtree.score(X_train,y_train))
print(dtree.score(X_test,y_test))


rds=RandomForestClassifier(criterion='gini')
print(rds.fit(X_train,y_train))
print(rds.score(X_train,y_train))
print(rds.score(X_test,y_test))

bg=BaggingClassifier(criterion='gini')
print(bg.fit(X_train,y_train))
print(bg.score(X_train,y_train))
print(bg.score(X_test,y_test))
