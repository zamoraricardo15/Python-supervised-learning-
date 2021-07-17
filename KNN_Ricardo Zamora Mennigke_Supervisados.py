# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:04:18 2020

@author: rzamoram
"""
#Tarea 1
#Ricardo Zamora Mennigke
#Metdoso Supervisados

##Pregunta 1: [25 puntos] Programe en lenguaje Python una funci´on que reciba como entrada la matriz de confusi´on 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
import seaborn as sns

def indices_general_MC(MC, nombres = None):
    precision_global = (np.sum(MC[0][0] + MC[1][1]) / np.sum(MC[0][0] + MC[1][1] + MC[1][0] + MC[0][1]))
    error_global = 1 - precision_global
    precision_categoria  = pd.DataFrame((MC[0][0] + MC[1][1])/np.sum(MC,axis = 1)).T
    precision_positiva = MC[1][1]/(MC[1][1] + MC[1][0])
    precision_negativa = MC[0][0]/(MC[0][0] + MC[0][1])
    falsos_positivos = MC[0][1]/(MC[0][0] + MC[0][1])
    falsos_negativos = MC[1][0]/(MC[1][0] + MC[1][1])
    asertividad_positiva = MC[1][1]/(MC[0][1] + MC[1][1])
    asertividad_negativa = MC[0][0]/(MC[0][0] + MC[1][0])
    if nombres!=None:
        precision_categoria.columns = nombres
    return {"Matriz de Confusión":MC, 
            "Precisión Global":precision_global, 
            "Error Global":error_global, 
            "Precisión por categoría":precision_categoria,
            "Precision Positiva (PP)": precision_positiva, 
            "Precision Negativa (PN)":precision_negativa, 
            "Falsos Positivos(FP)": falsos_positivos,
            "Falsos Negativos (FN)": falsos_negativos,
            "Asertividad Positiva (AP)": asertividad_positiva,
            "Asertividad Negativa (NP)": asertividad_negativa}

    
MC = [[892254, 212], [8993, 300]]
indices_general_MC(MC) 

##¿Es bueno o malo el modelo predictivo? Justifique su respuesta

#Inicialmente se puede entrar en el error de pensar que una precision global y un error global 
#tan acertados indican que el modelo es adecuado. Pero para un mejor analisis un criterio experto primero senalaria inicialmente 
#que existe una gran cantidad de falsos negativos, muy superior a los VP (verdaderos positivos), lo que indica que el modelo no 
#esta identificando correctamente los Reales positivos por lo que el modelo ajustado no es bueno.


#Pregunta 2: [25 puntos] En este ejercicio usaremos los datos (voces.csv). 


def indices_general(MC, nombres = None):
    precision_global = np.sum(MC.diagonal()) / np.sum(MC)
    error_global = 1 - precision_global
    precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
    precision_positiva = MC[1][1]/(MC[1][1] + MC[1][0])
    precision_negativa = MC[0][0]/(MC[0][0] + MC[0][1])
    falsos_positivos = 1 - precision_negativa
    falsos_negativos = 1 - precision_positiva
    asertividad_positiva = MC[1][1]/(MC[0][1] + MC[1][1])
    asertividad_negativa = MC[0][0]/(MC[0][0] + MC[1][0])
    if nombres!=None:
        precision_categoria.columns = nombres
    return {"Matriz de Confusión":MC, 
            "Precisión Global":precision_global, 
            "Error Global":error_global, 
            "Precisión por categoría":precision_categoria,
            "Precision Positiva (PP)": precision_positiva, 
            "Precision Negativa (PN)":precision_negativa, 
            "Falsos Positivos(FP)": falsos_positivos,
            "Falsos Negativos (FN)": falsos_negativos,
            "Asertividad Positiva (AP)": asertividad_positiva,
            "Asertividad Negativa (NP)": asertividad_negativa}

def distribucion_variable_predecir(data:DataFrame,variable_predict:str):
    colors = list(dict(**mcolors.CSS4_COLORS))
    df = pd.crosstab(index=data[variable_predict],columns="valor") / data[variable_predict].count()
    fig = plt.figure(figsize=(10,9))
    g = fig.add_subplot(111)
    countv = 0
    titulo = "Distribución de la variable %s" % variable_predict
    for i in range(df.shape[0]):
        g.barh(1,df.iloc[i],left = countv, align='center',color=colors[11+i],label= df.iloc[i].name)
        countv = countv + df.iloc[i]
    vals = g.get_xticks()
    g.set_xlim(0,1)
    g.set_yticklabels("")
    g.set_title(titulo)
    g.set_ylabel(variable_predict)
    g.set_xticklabels(['{:.0%}'.format(x) for x in vals])
    countv = 0 
    for v in df.iloc[:,0]:
        g.text(np.mean([countv,countv+v]) - 0.03, 1 , '{:.1%}'.format(v), color='black', fontweight='bold')
        countv = countv + v
    g.legend(loc='upper center', bbox_to_anchor=(1.08, 1), shadow=True, ncol=1)
    
def poder_predictivo_categorica(data:DataFrame, var:str, variable_predict:str):
    df = pd.crosstab(index= data[var],columns=data[variable_predict])
    df = df.div(df.sum(axis=1),axis=0)
    titulo = "Distribución de la variable %s según la variable %s" % (var,variable_predict)
    g = df.plot(kind='barh',stacked=True,legend = True, figsize = (10,9), \
                xlim = (0,1),title = titulo, width = 0.8)
    vals = g.get_xticks()
    g.set_xticklabels(['{:.0%}'.format(x) for x in vals])
    g.legend(loc='upper center', bbox_to_anchor=(1.08, 1), shadow=True, ncol=1)
    for bars in g.containers:
        plt.setp(bars, width=.9)
    for i in range(df.shape[0]):
        countv = 0 
        for v in df.iloc[i]:
            g.text(np.mean([countv,countv+v]) - 0.03, i , '{:.1%}'.format(v), color='black', fontweight='bold')
            countv = countv + v
            
def poder_predictivo_numerica(data:DataFrame, var:str, variable_predict:str):
    sns.FacetGrid(data, hue=variable_predict, height=6).map(sns.kdeplot, var, shade=True).add_legend()



pasada = os.getcwd()
os.chdir("C:/Users/rzamoram/Documents/Machine Learning/Métodos Supervisados con Python/Clase 01")
print(os.getcwd())
datos = pd.read_csv('voces.csv',delimiter=',',decimal=".")
print(datos.shape)
print(datos.head())
print(datos.info())

datos.describe(include = np.number)

corr = datos.corr()
print(corr)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)




poder_predictivo_numerica(datos,"meanfreq","genero")
poder_predictivo_numerica(datos,"sd","genero")
poder_predictivo_numerica(datos,"median","genero")
poder_predictivo_numerica(datos,"Q25","genero")
poder_predictivo_numerica(datos,"Q75","genero")
poder_predictivo_numerica(datos,"IQR","genero")
poder_predictivo_numerica(datos,"skew","genero")
poder_predictivo_numerica(datos,"kurt","genero")
poder_predictivo_numerica(datos,"sp.ent","genero")
poder_predictivo_numerica(datos,"sfm","genero")
poder_predictivo_numerica(datos,"mode","genero")
poder_predictivo_numerica(datos,"centroid","genero")
poder_predictivo_numerica(datos,"meanfun","genero")
poder_predictivo_numerica(datos,"minfun","genero")
poder_predictivo_numerica(datos,"maxfun","genero")
poder_predictivo_numerica(datos,"meandom","genero")
poder_predictivo_numerica(datos,"mindom","genero")
poder_predictivo_numerica(datos,"maxdom","genero")
poder_predictivo_numerica(datos,"dfrange","genero")
poder_predictivo_numerica(datos,"modindx","genero")


distribucion_variable_predecir(datos,"genero")

def indices_general_reducido(MC, nombres = None):
    precision_global = np.sum(MC.diagonal()) / np.sum(MC)
    error_global = 1 - precision_global
    precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
    if nombres!=None:
        precision_categoria.columns = nombres
    return {"Matriz de Confusión":MC, 
            "Precisión Global":precision_global, 
            "Error Global":error_global, 
            "Precisión por categoría":precision_categoria}

X = datos.iloc[:,:20] 
print(X.head())
y = datos.iloc[:,20:21] 
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
instancia_knn = KNeighborsClassifier(n_neighbors=3)
instancia_knn.fit(X_train,y_train.iloc[:,0].values)
print("Las predicciones en Testing son: {}".format(instancia_knn.predict(X_test)))
prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_reducido(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))

# Para ver los parámetros del modelo
KNeighborsClassifier()

instancia_knn = KNeighborsClassifier(n_neighbors=3,algorithm='brute',p=3)
instancia_knn.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))

#5. Repita el item d), pero esta vez, seleccione las 6 variables que, seg´un su criterio, tienen mejor poder predictivo. ¿Mejoran los resultados?

poder_predictivo_numerica(datos,"meanfreq","genero")
poder_predictivo_numerica(datos,"sd","genero")
poder_predictivo_numerica(datos,"median","genero")
poder_predictivo_numerica(datos,"Q25","genero")
poder_predictivo_numerica(datos,"Q75","genero")
poder_predictivo_numerica(datos,"IQR","genero")
poder_predictivo_numerica(datos,"skew","genero")
poder_predictivo_numerica(datos,"kurt","genero")
poder_predictivo_numerica(datos,"sp.ent","genero")
poder_predictivo_numerica(datos,"sfm","genero")
poder_predictivo_numerica(datos,"mode","genero")
poder_predictivo_numerica(datos,"centroid","genero")
poder_predictivo_numerica(datos,"meanfun","genero")
poder_predictivo_numerica(datos,"minfun","genero")
poder_predictivo_numerica(datos,"maxfun","genero")
poder_predictivo_numerica(datos,"meandom","genero")
poder_predictivo_numerica(datos,"mindom","genero")
poder_predictivo_numerica(datos,"maxdom","genero")
poder_predictivo_numerica(datos,"dfrange","genero")
poder_predictivo_numerica(datos,"modindx","genero")

X = datos.iloc[:,[1,3,5,8,9,12]] 
print(X.head())
y = datos.iloc[:,20:21]
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

instancia_knn = KNeighborsClassifier(n_neighbors=3)
instancia_knn.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))



A = np.matrix([[0.7413249211356467, 0.25867507886435326, 0.7717717717717718, 0.707641196013289, 0.292358803986711, 0.2282282282282282, 0.744927536231884, 0.7370242214532872],[0.9794952681388013, 0.02050473186119872, 0.975975975975976, 0.9833887043189369, 0.01661129568106312, 0.024024024024024038, 0.9848484848484849, 0.9736842105263158]])
mi_df = pd.DataFrame(A)
nombres_filas = ["K-vecinos todas variables","K-vecinos 6 variables"]
nombres_columnas = ["Precisi´on Global","Error Global","Precisi´on Positiva (PP)", "Precisi´on Negativa (PN)", "Falsos Positivos (FP)", "Falsos Negativos (FN)", "Asertividad Positiva (AP)", "Asertividad Negativa (AN)"]
mi_df = pd.DataFrame(A, index = nombres_filas, columns = nombres_columnas )
mi_df


#7. Repita el ejercicio 4, pero esta vez use en el m´etodo KNeighborsClassifier utilice los 4 diferentes algoritmos auto, ball tree, kd tree y brute. ¿Cu´al da mejores resultados? 
X = datos.iloc[:,:20] 
print(X.head())
y = datos.iloc[:,20:21] 
print(y.head()) 

# Para ver los parámetros del modelo
KNeighborsClassifier()

instancia_knn = KNeighborsClassifier(n_neighbors=3,algorithm='brute',p=3)
instancia_knn.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))  


KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=5, p=2,weights='uniform')
instancia_knn = KNeighborsClassifier(n_neighbors=3,algorithm='auto',p=3)
instancia_knn.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k]))) 
    
    
KNeighborsClassifier(algorithm='kd_tree', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=5, p=2,weights='uniform')
instancia_knn = KNeighborsClassifier(n_neighbors=3,algorithm='kd_tree',p=3)
instancia_knn.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k]))) 


KNeighborsClassifier(algorithm='ball_tree', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=5, p=2,weights='uniform')
instancia_knn = KNeighborsClassifier(n_neighbors=3,algorithm='ball_tree',p=3)
instancia_knn.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k]))) 
    
    
#Ejercicio 3: [25 puntos] Esta pregunta utiliza los datos (tumores.csv). Se trata de un conjunto de datos de caracter´ısticas 

pasada = os.getcwd()
os.chdir("C:/Users/rzamoram/Documents/Machine Learning/Métodos Supervisados con Python/Clase 01")
print(os.getcwd())
datos = pd.read_csv('tumores.csv',delimiter=',',decimal=".")
datos['imagen'] = datos['imagen'].astype('category')
print(datos.shape)
print(datos.head())
print(datos.info())

distribucion_variable_predecir(datos,"tipo")

X = datos.iloc[:,1:17] 
print(X.head())
y = datos.iloc[:,17:18] 
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=5, p=2,weights='uniform')
instancia_knn = KNeighborsClassifier(n_neighbors=3,algorithm='brute',p=3)
instancia_knn.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))  


KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=5, p=2,weights='uniform')
instancia_knn = KNeighborsClassifier(n_neighbors=3,algorithm='auto',p=3)
instancia_knn.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k]))) 
    
    
KNeighborsClassifier(algorithm='kd_tree', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=5, p=2,weights='uniform')
instancia_knn = KNeighborsClassifier(n_neighbors=3,algorithm='kd_tree',p=3)
instancia_knn.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k]))) 


KNeighborsClassifier(algorithm='ball_tree', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=5, p=2,weights='uniform')
instancia_knn = KNeighborsClassifier(n_neighbors=3,algorithm='ball_tree',p=3)
instancia_knn.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k]))) 
    

#Pregunta 4: [25 puntos] En este ejercicio vamos a predecir n´umeros escritos a mano (Hand Written Digit Recognition)

pasada = os.getcwd()
os.chdir("C:/Users/rzamoram/Documents/Machine Learning/Métodos Supervisados con Python/Clase 01")
print(os.getcwd())
datos = pd.read_csv('ZipDataTrainCod.csv',delimiter=';',decimal=".")
datos2 = pd.read_csv('ZipDataTestCod.csv',delimiter=';',decimal=".")

print(datos.shape)
print(datos.head())
print(datos.info())

print(datos2.shape)
print(datos2.head())
print(datos2.info())

X = datos.iloc[:,1:] 
print(X.head())
y = datos.iloc[:,0:1] 
print(y.head())

X2 = datos2.iloc[:,1:] 
print(X.head())
y2 = datos2.iloc[:,0:1] 
print(y.head())
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
instancia_knn = KNeighborsClassifier(n_neighbors=3)
instancia_knn.fit(X,y.iloc[:,0].values)
print("Las predicciones en Testing son: {}".format(instancia_knn.predict(X2)))
prediccion = instancia_knn.predict(X2)
MC = confusion_matrix(y2, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))



Numero = ['cero','uno', 'dos','cuatro','seis', 'siete','nueve']
datostrain358 = datos[~datos.Numero.isin(Numero)]
datostrain358.shape
datostrain358.head()
datostrain358.info()


Numero = ['cero','uno', 'dos','cuatro','seis', 'siete','nueve']
datostest358 = datos2[~datos2.Numero.isin(Numero)]
datostest358.shape
datostest358.head()
datostest358.info()


X = datostrain358.iloc[:,1:] 
print(X.head())
y = datostrain358.iloc[:,0:1] 
print(y.head())

X2 = datostest358.iloc[:,1:] 
print(X.head())
y2 = datostest358.iloc[:,0:1] 
print(y.head())
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
instancia_knn = KNeighborsClassifier(n_neighbors=3)
instancia_knn.fit(X,y.iloc[:,0].values)
print("Las predicciones en Testing son: {}".format(instancia_knn.predict(X2)))
prediccion = instancia_knn.predict(X2)
MC = confusion_matrix(y2, prediccion)
indices = indices_general_reducido(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))



#4. Repita los ejercicios 1, 2 y 3 pero reemplazando cada bloque 4 × 4 de p´ıxeles por su
import numpy as np
#p=4
#size=4
def block_matrix_four(size,X, p):
    ipix = X.shape[1] / (size^2)
    jpix = X.shape[0]
    pixmatrix = np.matrix(0,ipix, jpix)
    for i in range(1,ipix):
        for bl in range(1,size):
            pos = (bl-1)*size*p^2
            for h in range(1, size):
                s=0
                for p in range(1, size):
                    for k in range(1,size):
                        s = s+ X[i, pos+(h-1)*size + p^2*(p-1) + k]
                        new.matrix[i,h+(bl-1)*size] = s/size^2
    return(new.matrix)

def filter (X):
    xmatrix = np.matrix(X,16,16)
    fours = [1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,]
    X = X[fours = 1,] + X[fours = 2,] + X[fours = 3,] + X[fours = 4,]  
    X = X[,fours = 1] + X[,fours = 2] + X[,fours = 3,] + X[,fours = 4,]
    return(X/p^2)
    


