import pandas as pd
data=pd.read_csv('train.csv')
data2=pd.read_csv('test.csv')
columns=['Name','Ticket','Cabin']
data.drop(columns,axis=1,inplace=True)
data2.drop(columns,axis=1,inplace=True)
k,l=[],[]
a=data.pop('Embarked')
b=data2.pop('Embarked')
for i in a:
    if i=="S":
        k.append(1)
    elif i=="C":
        k.append(2)
    else:
        k.append(3)
for i in b:
    if i=="S":
        l.append(1)
    elif i=="C":
        l.append(2)
    else:
        l.append(3)
data['Embarked']=k
data2['Embarked']=l

sex=pd.get_dummies(data["Sex"], drop_first=False)
sex2=pd.get_dummies(data2["Sex"], drop_first=False)
data['Sex']=sex
data2['Sex']=sex2
data=data.interpolate(limit=4)
data["Age"]=data['Age'].round(0)
data2=data2.interpolate(limit=4)
data2["Age"]=data2['Age'].round(0)

pop=data.pop('Survived')

from sklearn.ensemble import RandomForestClassifier
f=RandomForestClassifier(n_estimators=50000)
f.fit(data,pop)
pred=f.predict(data2)    
from pandas import DataFrame
submission=pd.DataFrame({"PassengerId":data2['PassengerId'],
                         "Survived":pred})
    
    
submission.to_csv("Shreya.csv",index=False)