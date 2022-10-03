#!/usr/bin/env python
# coding: utf-8
Car Evaluation Data Set

# In[386]:


pip install matplotlib


# In[387]:


import pandas as pd
import matplotlib.pyplot as plt


# In[388]:


donnees = pd.read_csv("CAR EVALUATION DATASET/car.data",sep=",",names = ["buying","maint","doors","persons","lug_boot","safety","target"])


# In[389]:


donnees.info()


# In[390]:


donnees.describe()


# In[391]:


donnees = donnees.dropna()
y = donnees.drop(["buying", "maint", "doors", "persons", "lug_boot", "safety"],axis = 1)
X = donnees.drop(["target"], axis=1)


# In[392]:


X.info()


# In[393]:


print(y.info())
print(y)


# In[394]:


#Verifier les différentes valeurs du dataset :

def valeurUnique(y, colonne):
    valeur = []
    for i in range(len(y)):
        if y.iloc[i,colonne] not in valeur:
            valeur.append(y.iloc[i,colonne])
    return valeur

valeurUnique(y,0)


# In[395]:


#Attribuer des valeurs numériques aux données du dataset (target)

unacc = 0
acc = 0
good = 0
vgood = 0

for i in range(len(y)):
    if y.iloc[i,0] == "unacc":
        y.iloc[i,0] = 0
        unacc+=1
    elif y.iloc[i,0] == "acc":
        y.iloc[i,0] = 1
        acc+=1
    elif y.iloc[i,0] == "good":
        y.iloc[i,0] = 2
        good+=1
    elif y.iloc[i,0] == "vgood":
        y.iloc[i,0] = 3
        vgood+=1

valeurUnique(y,0)
print("Unacc :",unacc,"\nacc : ",acc,"\ngood : ",good,"\nvgood : ", vgood)


# In[396]:


#Attribuer des valeurs numériques aux données du dataset X

#valeurUnique(X,0) #valeurs pour la colonne Buying


def attribution(X,nombreColonnes):
    for i in range(nombreColonnes):
        liste= valeurUnique(X, i)
        nomColonne = X.columns[i]
        for j in range(len(X)):
            if X.iloc[j,i] in liste:
                X.iloc[j,i] = liste.index(X.iloc[j,i])
        donnees[nomColonne].value_counts(normalize=True).plot(kind='pie', autopct="%1.1f%%")
        plt.show()
        
attribution(X,len(X.axes[1]))


# In[397]:


X


# In[398]:


from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV


# In[399]:


#création de l'arbre de décision
X_train, X_test, Y_train, Y_test = train_test_split(X, y.astype('int'), train_size=0.7, random_state=0)
clf = tree.DecisionTreeClassifier(max_depth = 3, min_samples_leaf=60)
clf.fit(X_train, Y_train)
tree.plot_tree(clf, filled=True)
print(clf.score(X_test, Y_test))


# In[400]:


#on fait un gridSearch
tuned_params = {"max_depth":[1, 2, 3, 4, 5, 6, 10, 20, 30], "min_samples_leaf":[1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]}
grid_search = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=tuned_params, cv=10)
grid_search.fit(X_train, Y_train)
print(grid_search.best_estimator_.score(X_test, Y_test))
print(grid_search.best_params_)


# In[401]:


clf = tree.DecisionTreeClassifier(max_depth = 10, min_samples_leaf=1)
clf.fit(X_train, Y_train)

tree.plot_tree(clf, filled=True, fontsize=5)
print(clf.score(X_test, Y_test))


# In[402]:


donnees.info()


# In[403]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y.astype('int'), train_size=0.95, random_state=0)

#on fait un gridSearch
tuned_params = {"max_depth":[1, 2, 3, 4, 5, 6, 10, 20, 30], "min_samples_leaf":[1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]}
grid_search = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=tuned_params, cv=10)
grid_search.fit(X_train, Y_train)
print(grid_search.best_estimator_.score(X_test, Y_test))
print(grid_search.best_params_)



# In[404]:



clf = tree.DecisionTreeClassifier(max_depth = 10, min_samples_leaf=1)
clf.fit(X_train, Y_train)

tree.plot_tree(clf, filled=True, fontsize=2)
print(clf.score(X_test, Y_test))


# In[405]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


resultat = clf.predict(X_test)
clf.score(X_test, Y_test)
#score en test de l'arbre de décision 0.9777777777777777
print(clf.score(X_test, Y_test))
cm =confusion_matrix(resultat, Y_test)
print(confusion_matrix(resultat, Y_test))


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.show()


# In[406]:


import numpy as np

# Paramètres
n_classes = 4
plot_colors = "bry" 
# blue-red-yellow
plot_step = 0.02
# Choisir les attributs longueur et largeur des pétales
pair = [2, 3]
# On ne garde seulement les deux attributs
X2 = X.iloc[:, pair]
X2 = X2.to_numpy()
y = y.to_numpy()

# Apprentissage de l'arbre
clf = tree.DecisionTreeClassifier().fit(X2, y.astype('int'))
# Affichage de la surface de décision
x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(1, 5, plot_step), np.arange(1, 4, plot_step))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

plt.axis("tight")
# Affichage des points d'apprentissage
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    plt.scatter(X2[idx, 0], X2[idx, 1], c=color, label=y[i],cmap=plt.cm.Paired)
plt.axis("tight")
plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend()
plt.show()

