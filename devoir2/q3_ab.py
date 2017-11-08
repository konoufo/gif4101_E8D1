import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


##### Importer les données du problemes  ############################
features = open("features.txt","r")


X = np.loadtxt("data.txt", delimiter=',')
y = np.loadtxt("target.txt", delimiter=',')
features1 = features.read().split()

#### partition ######

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify =y, test_size=0.5, random_state = 42)


###### 3a)  ####################

chi = SelectKBest(chi2,k=10)
varselect = chi.fit(X_train,y_train).get_support(indices=True)
xtrain_new = chi.transform(X_train)
xtest_new = chi.transform(X_test)
word1 = [features1[x] for x in varselect]




kb = SelectKBest(mutual_info_classif,k=10)
varselect2 = kb.fit(X_train,y_train).get_support(indices=True)
xtrain_new2 = kb.transform(X_train)
xtest_new2 = kb.transform(X_test)
word2 = [features1[x] for x in varselect2]


model = LinearSVC()
score_chi2 = model.fit(xtrain_new,y_train).score(xtest_new,y_test)
score_mutual = model.fit(xtrain_new2,y_train).score(xtest_new2,y_test)
score_total = model.fit(X_train,y_train).score(X_test,y_test)
print('test chi2')
print(score_chi2)
print(word1)

print('test mutual')
print(score_mutual)
print(word2)

print('test total')
print(score_total)


######## 3b) ################

RFE = RFE(model,10,step=1)
varselect3 = RFE.fit(X_train,y_train).get_support(indices=True)
xtrain_rfe = RFE.transform(X_train)
xtest_rfe = RFE.transform(X_test)
wordrfe = [features1[x] for x in varselect3]
score_rfe = model.fit(xtrain_rfe,y_train).score(xtest_rfe,y_test)
print('test_rfe')
print(score_rfe)
print(wordrfe)

