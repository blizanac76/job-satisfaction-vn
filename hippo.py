import numpy as np

import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif

#iz nekog razloga morao sam mu dati celu putanju do csv fajla
df = pd.read_csv('C:/Users/Blizanac/Desktop/Projekaaat/New folder/EmployeeSatisfactionIndex.csv')

# nepotrebne kolone
df = df.drop(['emp_id', 'location'], axis=1)  

# odbacivanje pocetne kolone iz csv
df2 = df.drop(['Unnamed: 0'], axis=1)

# Drop non-numeric columns
df2 = df2.select_dtypes(include=[np.number])

#string u broj
df2['education'] = pd.factorize(df['education'])[0]  
df2['recruitment_type'] = pd.factorize(df['recruitment_type'])[0]
df2['satisfied'] = df['satisfied'].astype(int)



# mapa koliko sta utice na poslovnu satisfakciju
plt.figure(figsize=(10, 7))
sns.heatmap(df2.corr(), annot=True, cmap='coolwarm', center=.35)
plt.title('Povezanost zadovoljstva posla', fontsize=10)
plt.show()

# zadovoljstvo i godine

sns.boxenplot(x='satisfied', y='age', data=df)

# Show the plot
plt.show()
##############
# prevodjenje u broj ostalih stringova
df = df2.replace(['HR', 'Technology', 'Sales', 'Purchasing', 'Marketing'], [1, 2, 3, 4, 5])
df = df2.replace(['Suburb', 'City'], [1, 2])
df = df2.replace(['PG', 'UG'], [1, 2])
df = df2.replace(['Referral', 'Walk-in', 'On-Campus', 'Recruitment Agency'], [1, 2, 3, 4])

###################

X = df.drop(['satisfied'], axis=1)
y = df['satisfied']

# Train/test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)


models = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    LogisticRegression()
]
#posto diploma ne utice dobro tu kolonu izbacujem
df_cisto = df.drop(columns=['certifications'], axis=1)

#  'age' u numeric
df_cisto['age'] = pd.to_numeric(df['age'], errors='coerce')

#haetmap za ciste podatke, bez diplome
plt.figure(figsize=(10, 7))
sns.heatmap(df_cisto.corr(), cmap='coolwarm', center=.1, annot=True)
plt.show()


#histogram rejtinga po godinama
plt.figure(figsize=(10, 7))
sns.boxplot(data=df_cisto, x='rating', y='age', palette='muted')
plt.title('Ocena po godinama', fontsize=10)
plt.xlabel('ocena', fontsize=10)
plt.ylabel('godine', fontsize=10)
plt.show()



for model in models:
    name = model.__class__.__name__
    print("\n*****************\n")
    
    print(name)
    print("\n*****************\n")
    
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    
    print("Accuracy: ", accuracy_score(y_test, y_predict))
    print("Precision: ", precision_score(y_test, y_predict))
    print("Recall: ", recall_score(y_test, y_predict))
    print("F1 Score: ", f1_score(y_test, y_predict))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_predict))
    print()
    
    # Crossvalidacija
    scores = cross_val_score(model, X, y, cv=10)
    print("crossvalidation: ", scores)
    print("average crossvalidation: ", scores.mean())
    print()
    
    selector = SelectKBest(score_func=f_classif, k=5)  # selekcija 5 najboljih
    #trening po njima
    selector.fit(X_train, y_train)
    
    
    selected_features = X.columns[selector.get_support()]
    
    # novi setovi za krosvalidaciju
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    model.fit(X_train_selected, y_train)
    y_predict_selected = model.predict(X_test_selected)
    
    print("Accuracy : ", accuracy_score(y_test, y_predict_selected))
    print("Precision : ", precision_score(y_test, y_predict_selected))
    print("Recall : ", recall_score(y_test, y_predict_selected))
    print("F1 Score : ", f1_score(y_test, y_predict_selected))
    print("Confusion Matrix :")
    
   #
    print(confusion_matrix(y_test, y_predict_selected))
   
    """
    model.fit(X1_train, y1_train)
    y1_predict = model.predict(X1_test)
    print("Accuracy: ", accuracy_score(y1_test, y1_predict))
    print("Precision: ", precision_score(y1_test, y1_predict))
    print("Recall: ", recall_score(y1_test, y1_predict))
    print("F1 Score: ", f1_score(y1_test, y1_predict))
    print("Confusion Matrix:")
    print(confusion_matrix(y1_test, y1_predict))
    print()
    scores = cross_val_score(model, X1, y1, cv=5)
    print("crossvalidacija uspeh: ", scores)
    print("srednji Score krosvalidacije: ", scores.mean())
    print()

    selector = SelectKBest(score_func=f_classif, k=5)  
    selector.fit(X1_train, y1_train)
    selected_features = X1.columns[selector.get_support()]
    X1_train_selected = selector.transform(X1_train)
    X1_test_selected = selector.transform(X1_test)
    model.fit(X1_train_selected, y1_train)
    y1_predict_selected = model.predict(X1_test_selected)
    print("Accuracy : ", accuracy_score(y1_test, y1_predict_selected))
    print("Precision : ", precision_score(y1_test, y1_predict_selected))
    print("Recall : ", recall_score(y1_test, y1_predict_selected))
    print("F1 Score : ", f1_score(y1_test, y1_predict_selected))
    print("Confusion Matrix :")
    print(confusion_matrix(y1_test, y1_predict_selected))
    """
