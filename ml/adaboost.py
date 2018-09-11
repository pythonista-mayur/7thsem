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
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
cart = DecisionTreeClassifier()
model = BaggingClassifier(base_estimator=cart,n_estimators=100,random_state=99)
score = cross_val_score(model,X,y,cv=10)
print(f"final accuracy after cross validation:{score.mean()}")