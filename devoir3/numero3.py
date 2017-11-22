import numpy as np
from sklearn import datasets
from collections import Counter
import random 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

##########  Edition de Wilson  ###################

def edit_wilson(x,r,k):
    
    k = k+1  #k+1 car on élimine par la suite le point étudié
    classe = np.empty(k-1)
    selection = np.ones(x.shape[0])
    c = np.arange(x.shape[0])
    z = np.arange(x.shape[0])
    np.random.shuffle(c)
    for i in c:
        neigh = KNeighborsClassifier(n_neighbors=k)
        t = r[z]   # sélectionner uniquement classe des données qui sont dans z
        if len(t) < k:  # gestion du cas ou il y a moins de données que de k-ppv
            z1 = z[z != i]
            nn = z1
        else:
            neigh.fit(x[z],t)  # fit sur données de z
            nn = np.array(neigh.kneighbors(x[i], return_distance=False))[np.newaxis]
            nn = np.delete(nn,0)  #élimination du ppv correspondant au point x[i]
        for e in range(0,nn.shape[0]):
            cla = r[nn[e]]
            classe[e] = cla
        count = Counter(classe).most_common(1)  
        count = np.asarray(count)
        if r[i] == count[:,0]:
            selection[i] = 1
        else:
            selection[i] = 0
            index = np.argwhere(z == i)
            z = np.delete(z,index)
            
    return z


############  Condensation de Hart  #####################

def condense_hart(x,r):
    
    r1 = np.arange(x.shape[0])
    r1 = np.concatenate([r1,r])
    r1 = np.reshape(r1,(2,x.shape[0])).T  #pour conserver indice de position, 


    selection = np.zeros(x.shape[0])
    selection1 = np.ones(x.shape[0])

    z = np.int32(random.randint(0,x.shape[0]-1))
    while (selection != selection1).all(): 
        selection = selection1
        c = np.arange(x.shape[0])
        np.random.shuffle(c)
        for i in c:
            if z.shape == ():
                nn = z # cas ou z est vide
            else:
                neigh = KNeighborsClassifier(n_neighbors=1)
                x1 = x[z]
                t = r1[z]
                neigh.fit(x1,t[:,1])
                nn = neigh.kneighbors(x[i], return_distance=False)
                nn = t[nn,0]
            if r[i] == r[nn]: 
                selection1[i] = 1 
            else:
                selection1[i] = 0
                z = np.append(z,i)
          

    z = np.sort(z)  
    return z

############## Numero 3c - test sur iris de Fisher  ###########################

##########Gestion des données ##########################
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_name = iris.target_names[:3]

## normalisation
#X1 = np.divide((X - X.min()), (X.max()-X.min()))

mini = X.min(axis=0)
maxi = X.max(axis=0)

X[:,0] = np.divide((X[:,0] - mini[0]), (maxi[0]-mini[0]))
X[:,1] = np.divide((X[:,1] - mini[1]), (maxi[1]-mini[1]))
X[:,2] = np.divide((X[:,2] - mini[2]), (maxi[2]-mini[2]))
X[:,3] = np.divide((X[:,3] - mini[3]), (maxi[3]-mini[3]))


   
x1, x3, y1, y3 = train_test_split(X,y,test_size=0.33,stratify=y)

x1, x2, y1, y2 = train_test_split(x1,y1,test_size=0.5,stratify=y1)


##### PLI 1 ###############
p1 = np.append(x1,x2)
p1 = np.reshape(p1,(100,4))
py1 = np.append(y1, y2)
test_x1 = x3
test_y1 = y3

######## PLI 2 #############

p2 = np.append(x1, x3)
p2 = np.reshape(p2,(100,4))
py2 = np.append(y1,y3)
test_x2 = x2
test_y2 = y2

########### PLI 3 ############

p3 = np.append(x2, x3)
p3 = np.reshape(p2, (100,4))
py3 = np.append(y2,y3)
test_x3 = x1
test_y3 = y1


#########  ÉVALUATION edit wilson ###########################################
tsel = np.zeros((3,5))
perf = np.zeros((3,5))

kn = [1,3,5,11,23]
e = 0
for i in kn:
    z = edit_wilson(p1,py1,i)
    tsel[0,e] = z.shape[0]/p1.shape[0]
    p1a = p1[z]
    py1a = py1[z]
    neigh = KNeighborsClassifier(n_neighbors = i)
    perf[0,e] = neigh.fit(p1a,py1a).score(test_x1,test_y1)

    z = edit_wilson(p2,py2,i)
    tsel[1,e] = z.shape[0]/p2.shape[0]
    p2a = p2[z]
    py2a = py2[z]
    neigh = KNeighborsClassifier(n_neighbors = i)
    perf[1,e] = neigh.fit(p2a,py2a).score(test_x2,test_y2)

    z = edit_wilson(p3,py3,i)
    tsel[2,e] = z.shape[0]/p3.shape[0]
    p3a = p3[z]
    py3a = py3[z]
    neigh = KNeighborsClassifier(n_neighbors = i)
    perf[2,e] = neigh.fit(p3a,py3a).score(test_x3,test_y3)
    e = e + 1
    
    
#########  ÉVALUATION condensat hart ###########################################
tselhart = np.zeros((3,5))
perfhart = np.zeros((3,5))

kn = [1,3,5,11,23]
e = 0
for i in kn:
    z = condense_hart(p1,py1)
    tselhart[0,e] = z.shape[0]/p1.shape[0]
    if z.shape[0] < i:  #gestion du cas ou il y a moins de données dans z que de k-ppv
        k = z.shape[0]
    else:
        k = i
    p1a = p1[z]
    py1a = py1[z]
    neigh = KNeighborsClassifier(n_neighbors = k)
    perfhart[0,e] = neigh.fit(p1a,py1a).score(test_x1,test_y1)

    z = condense_hart(p2,py2)
    tselhart[1,e] = z.shape[0]/p2.shape[0]
    if z.shape[0] < i:
        k = z.shape[0]
    else:
        k = i
    p2a = p2[z]
    py2a = py2[z]
    neigh = KNeighborsClassifier(n_neighbors =k)
    perfhart[1,e] = neigh.fit(p2a,py2a).score(test_x2,test_y2)

    z = condense_hart(p3,py3)
    tselhart[2,e] = z.shape[0]/p3.shape[0]
    if z.shape[0] < i:
        k = z.shape[0]
    else:
        k = i
    p3a = p3[z]
    py3a = py3[z]
    neigh = KNeighborsClassifier(n_neighbors = k)
    perfhart[2,e] = neigh.fit(p3a,py3a).score(test_x3,test_y3)
    e = e + 1
    
    z1=z
    
#########  ÉVALUATION hart + wilson ###########################################
tselcomb = np.zeros((3,5))
perfcomb = np.zeros((3,5))

kn = [1,3,5,11,23]
e = 0
for i in kn:
    z = edit_wilson(p1,py1,i)
    p1a = p1[z]
    py1a = py1[z]
    z = condense_hart(p1a,py1a)
    tselcomb[0,e] = z.shape[0]/p1.shape[0]
    if z.shape[0] < i:
        k = z.shape[0]
    else:
        k = i
    p1a = p1a[z]
    py1a = py1a[z]
    neigh = KNeighborsClassifier(n_neighbors = k)
    perfcomb[0,e] = neigh.fit(p1a,py1a).score(test_x1,test_y1)

    z = edit_wilson(p2,py2,i)
    p2a = p2[z]
    py2a = py2[z]
    z = condense_hart(p2a,py2a)
    tselcomb[1,e] = z.shape[0]/p2.shape[0]
    if z.shape[0] < i:
        k = z.shape[0]
    else:
        k = i
    p2a = p2a[z]
    py2a = py2a[z]
    neigh = KNeighborsClassifier(n_neighbors =k)
    perfcomb[1,e] = neigh.fit(p2a,py2a).score(test_x2,test_y2)

    z = edit_wilson(p3,py3,i)
    p3a = p3[z]
    py3a = py3[z]
    z = condense_hart(p3a,py3a)
    tselcomb[2,e] = z.shape[0]/p3.shape[0]
    if z.shape[0] < i:
        k = z.shape[0]
    else:
        k = i
    p3a = p3a[z]
    py3a = py3a[z]
    neigh = KNeighborsClassifier(n_neighbors = k)
    perfcomb[2,e] = neigh.fit(p3a,py3a).score(test_x3,test_y3)
    e = e + 1
    
    
################# printing results ####################

tsel_mean = np.mean(tsel,axis=0)
tselhart_mean = np.mean(tselhart, axis=0)
tselcomb_mean = np.mean(tselcomb, axis=0)

perf_mean = np.mean(perf, axis=0)
perfhart_mean = np.mean(perfhart, axis=0)
perfcomb_mean = np.mean(perfcomb, axis=0)

print('le taux de sélection moyen pour les 3 prototypes en fonction de k est:')
print(tsel_mean)
print(tselhart_mean)
print(tselcomb_mean)

print('l''erreur moyenne pour les 3 prototypes selon k est:')
print(1-perf_mean)
print(1-perfhart_mean)
print(1-perfcomb_mean)

plt.plot(kn, perf_mean,'r--',label = 'Wilson' )        
plt.plot(kn, perfhart_mean,'g--',label = 'Hart' ) 
plt.plot(kn, perfcomb_mean,'b--',label = 'combiné' )  
plt.legend(loc=1,
     fontsize=8)
plt.xlabel('Nombres de k-voisns')
plt.ylabel('Performance')
plt.show()       
        
