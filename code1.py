#Imports

#pandas
import pandas as pd
from pandas import Series, DataFrame

#numpy, matplotlib
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
#% matplotlib inline

# machine learning classifiers 

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


#get data

mobileTrain = pd.read_csv("train.csv")
mobileTest = pd.read_csv("test.csv")


#mobileTrain['sc'] = mobileTrain['sc_h']*mobileTrain['sc_w']
#mobileTrain['cam'] = mobileTrain['pc']
#mobileTrain['internet'] = mobileTrain['three_g']
#mobileTrain['px'] = mobileTrain['px_width']*mobileTrain['px_height']


#mobileTest['sc'] = mobileTest['sc_h']*mobileTest['sc_w']
#mobileTest['cam'] = mobileTest['pc']
#mobileTest['internet'] = mobileTest['three_g']
#mobileTest['px'] = mobileTest['px_width']*mobileTest['px_height']
'''
dropElements = ['m_dep','n_cores']
mobileTrain = mobileTrain.drop(dropElements,axis = 1)
mobileTest = mobileTest.drop(dropElements,axis = 1)
'''

id1 = mobileTest["id"]
mobileTest = mobileTest.drop("id",axis=1)

#head of data

#print mobileTest.head()
#print mobileTrain.head()
#print mobileTrain.describe()
#print mobileTrain.describe(include='all')
#mobileTrain.info()
mobileTrain1 = mobileTrain[:1300]
mobileTrain2 = mobileTrain[1300:]

#print mobileTrain2.info()



#correlation heatmap
'''
colormap = plt.cm.viridis
plt.figure(figsize = (12,12))
plt.title('Correlation of Features', y = 1.05, size = 15)
sns.heatmap(mobileTrain1.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
'''





X_train = mobileTrain1.drop("price_range",axis = 1)
Y_train = mobileTrain1["price_range"]

X_test = mobileTrain2.drop("price_range",axis = 1)



Y_test  = mobileTrain2["price_range"]


'''

def plot_variable_importance( X , y ):
    tree = DecisionTreeClassifier( random_state = 99 )
    tree.fit( X , y )
    plot_model_var_imp( tree , X , y )
    
    
def plot_model_var_imp( model , X , y ):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = X.columns 
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[ : 20].plot( kind = 'barh' )
    print (model.score( X , y ))
    
plot_variable_importance(X_train,Y_train)

'''


#
# logistic regression
'''
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

print logreg.score(X_test, Y_test)
'''

'''
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

print random_forest.score(X_test, Y_test)
'''


'''
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

print knn.score(X_test, Y_test)

'''




svc = LinearSVC()
svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

print svc.score(X_test, Y_test)


'''
gnb = GaussianNB()
gnb.fit(X_train, Y_train)

Y_pred = gnb.predict(mobileTest)

print gnb.score(X_test, Y_test)
'''
'''
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)

Y_pred = dtc.predict(mobileTest)

print dtc.score(X_test, Y_test)
'''

'''

StackingSubmission = pd.DataFrame({'id':id1,'price_range': Y_pred })
StackingSubmission.to_csv("StackingSubmission.csv", index=False)
'''
