import pandas as pd
from sklearn.model_selection import train_test_split



df = pd.read_csv("wine.csv",header=None)

df.columns = ["class","Alcohol","Malic_acid","Ash","Alcalinity_of_ash","Magnesium",
              "Total_phenols","Flavanoids","Nonflavanoid_phenols",
              "Proanthocyanins","Color_intensity","Hue","OD280/OD315_of_diluted_wines",
              "Proline",]
X = df.drop('class',axis=1)
y = df['class']
X_train, X_test, Y_train,Y_test = train_test_split(X,y,test_size=0.33,random_state=99)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train)
rfc_predict =  rfc.predict(X_test)

from sklearn.model_selection import cross_val_score
rfc_cv_score = cross_val_score(rfc,X,y,cv=10)

from sklearn.metrics import confusion_matrix,classification_report


print("=== Confusion Matrix ===")
print(confusion_matrix(Y_test, rfc_predict))
print('\n')

print("=== Classification Report ===")
print(classification_report(Y_test, rfc_predict))
print('\n')

print("=== All AUC Scores ===")

print(rfc_cv_score)
print('\n')

print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())