# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 03:01:56 2020

@author: rzamoram
"""

#Pregunta 2: [25 puntos] En este ejercicio usaremos los datos (voces.csv). 

#1. Cargue la tabla de datos voces.csv en Python.
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import graphviz # Debe instalar este paquete 
import pandas as pd
import matplotlib.image as mpimg
from matplotlib import colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from pandas import DataFrame


pasada = os.getcwd()
os.chdir("C:/Users/rzamoram/Documents/Machine Learning/Métodos Supervisados con Python/Clase 01")
print(os.getcwd())
datos = pd.read_csv('voces.csv',delimiter=',',decimal=".")
print(datos.shape)
print(datos.head())
print(datos.info())

datos.describe(include = np.number)

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

distribucion_variable_predecir(datos,"genero")

#2. Use Arboles de Decisi´on en ´ Python (con los par´ametros por defecto) para generar un
X = datos.iloc[:,:20] 
print(X.head())
y = datos.iloc[:,20:21] 
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

instancia_arbol = DecisionTreeClassifier(random_state=0)

instancia_arbol.fit(X_train,y_train)

print("Las predicciones en Testing son: {}".format(instancia_arbol.predict(X_test)))

prediccion = instancia_arbol.predict(X_test)
MC = confusion_matrix(y_test, prediccion)


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
    
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


#3. Usando la funci´on programada en el ejercicio 1 de la tarea anterior, los datos voces.csv
A = np.matrix([[0.7413249211356467, 0.25867507886435326, 0.7717717717717718, 0.707641196013289, 0.292358803986711, 0.2282282282282282, 0.744927536231884, 0.7370242214532872],[0.9794952681388013, 0.02050473186119872, 0.975975975975976, 0.9833887043189369, 0.01661129568106312, 0.024024024024024038, 0.9848484848484849, 0.9736842105263158]])
mi_df = pd.DataFrame(A)
nombres_filas = ["K-vecinos todas variables","K-vecinos 6 variables"]
nombres_columnas = ["Precisi´on Global","Error Global","Precisi´on Positiva (PP)", "Precisi´on Negativa (PN)", "Falsos Positivos (FP)", "Falsos Negativos (FN)", "Asertividad Positiva (AP)", "Asertividad Negativa (AN)"]
mi_df = pd.DataFrame(A, index = nombres_filas, columns = nombres_columnas )
mi_df

###FALTA


#4. Grafique el ´arbol generado e interprete al menos dos reglas que se puedan extraer del

    
    
def graficar_arbol(grafico = None):
    grafico.format = "png"
    archivo  = grafico.render()
    img = mpimg.imread(archivo)
    imgplot = plt.imshow(img)
    plt.axis('off')
    
datos_plotear = export_graphviz(instancia_arbol, out_file=None,class_names=["Masculino", "Femanino"],
                feature_names=list(X.columns.values), filled=True)
grafico = graphviz.Source(datos_plotear) 
plt.rcParams['figure.figsize'] = [15, 15] # Tamaño del gráfico
graficar_arbol(grafico)  

##Podar
instancia_arbol2 = DecisionTreeClassifier(min_samples_leaf=150)
instancia_arbol2.fit(X_train,y_train)
prediccion = instancia_arbol2.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


datos_plotear = export_graphviz(instancia_arbol2, out_file=None,class_names=["Masculino", "Femanino"],
                feature_names=list(X.columns.values), filled=True)
grafico = graphviz.Source(datos_plotear) 
plt.rcParams['figure.figsize'] = [15, 15] # Tamaño del gráfico
graficar_arbol(grafico)


#5. Repita los ejercicios 1-4, pero esta vez use 2 combinaciones diferentes de los par´ametros
##max depth = 2 y criteration=entropy
instancia_arbol = DecisionTreeClassifier(criterion='entropy',max_depth=2)
instancia_arbol.fit(X_train,y_train)
print("Las predicciones en Testing son: {}".format(instancia_arbol.predict(X_test)))

prediccion = instancia_arbol.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))
    
datos_plotear = export_graphviz(instancia_arbol, out_file=None,class_names=["Masculino", "Femenino"],
                feature_names=list(X.columns.values), filled=True)
grafico = graphviz.Source(datos_plotear) 
plt.rcParams['figure.figsize'] = [15, 15] # Tamaño del gráfico
graficar_arbol(grafico) 

##max depth = 2 y criteration=entropy y splitter=random
instancia_arbol = DecisionTreeClassifier(criterion='gini', splitter="random", max_depth=3)
instancia_arbol.fit(X_train,y_train)
print("Las predicciones en Testing son: {}".format(instancia_arbol.predict(X_test)))

prediccion = instancia_arbol.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))
    
datos_plotear = export_graphviz(instancia_arbol, out_file=None,class_names=["Masculino", "Femenino"],
                feature_names=list(X.columns.values), filled=True)
grafico = graphviz.Source(datos_plotear) 
plt.rcParams['figure.figsize'] = [15, 15] # Tamaño del gráfico
graficar_arbol(grafico) 


A = np.matrix([[0.7413249211356467, 0.25867507886435326, 0.7717717717717718, 0.707641196013289, 0.292358803986711, 0.2282282282282282, 0.744927536231884, 0.7370242214532872],
               [0.9794952681388013, 0.02050473186119872, 0.975975975975976, 0.9833887043189369, 0.01661129568106312, 0.024024024024024038, 0.9848484848484849, 0.9736842105263158],
              [0.9605678233438486, 0.039432176656151396, 0.963963963963964, 0.9568106312292359, 0.04318936877076407, 0.036036036036036, 0.9610778443113772, 0.96],
              [0.944794952681388, 0.05520504731861198, 0.948948948948949, 0.9401993355481728, 0.05980066445182719, 0.05105105105105101, 0.9461077844311377, 0.9433333333333334],
              [0.9006309148264984, 0.09936908517350163, 0.9519519519519519, 0.8438538205980066, 0.15614617940199338, 0.048048048048048075, 0.8708791208791209, 0.9407407407407408]])
mi_df = pd.DataFrame(A)
nombres_filas = ["K-vecinos todas variables","K-vecinos 6 variables","Arbol de Decision (default)","Arbol de Decision (criterion='entropy',max_depth=2)","Arbol de Decision (criterion='gini', splitter='random', max_depth=3)"]
nombres_columnas = ["Precisi´on Global","Error Global","Precisi´on Positiva (PP)", "Precisi´on Negativa (PN)", "Falsos Positivos (FP)", "Falsos Negativos (FN)", "Asertividad Positiva (AP)", "Asertividad Negativa (AN)"]
mi_df = pd.DataFrame(A, index = nombres_filas, columns = nombres_columnas )
mi_df

#6. Repita los ejercicios 1-4, pero esta vez use 2 combinaciones diferentes de selecci´on de 6
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


instancia_arbol3 = DecisionTreeClassifier()
instancia_arbol3.fit(X_train,y_train)
prediccion = instancia_arbol3.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


datos_plotear = export_graphviz(instancia_arbol3, out_file=None,class_names=["Masculino", "Femenino"],
                feature_names=list(X.columns.values), filled=True)
grafico = graphviz.Source(datos_plotear) 
plt.rcParams['figure.figsize'] = [20, 20] # Tamaño del gráfico
graficar_arbol(grafico)


##Podar
instancia_arbol2 = DecisionTreeClassifier(min_samples_leaf=150)
instancia_arbol2.fit(X_train,y_train)
prediccion = instancia_arbol2.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


datos_plotear = export_graphviz(instancia_arbol2, out_file=None,class_names=["Masculino", "Femanino"],
                feature_names=list(X.columns.values), filled=True)
grafico = graphviz.Source(datos_plotear) 
plt.rcParams['figure.figsize'] = [15, 15] # Tamaño del gráfico
graficar_arbol(grafico)

###Ahora con otras combinaciones de parametros
instancia_arbol = DecisionTreeClassifier(criterion='entropy',max_depth=2)
instancia_arbol.fit(X_train,y_train)
print("Las predicciones en Testing son: {}".format(instancia_arbol.predict(X_test)))

prediccion = instancia_arbol.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))
    
datos_plotear = export_graphviz(instancia_arbol, out_file=None,class_names=["Masculino", "Femenino"],
                feature_names=list(X.columns.values), filled=True)
grafico = graphviz.Source(datos_plotear) 
plt.rcParams['figure.figsize'] = [15, 15] # Tamaño del gráfico
graficar_arbol(grafico) 

##max depth = 2 y criteration=entropy y splitter=random
instancia_arbol = DecisionTreeClassifier(criterion='gini', splitter="random", max_depth=3)
instancia_arbol.fit(X_train,y_train)
print("Las predicciones en Testing son: {}".format(instancia_arbol.predict(X_test)))

prediccion = instancia_arbol.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))
    
datos_plotear = export_graphviz(instancia_arbol, out_file=None,class_names=["Masculino", "Femenino"],
                feature_names=list(X.columns.values), filled=True)
grafico = graphviz.Source(datos_plotear) 
plt.rcParams['figure.figsize'] = [15, 15] # Tamaño del gráfico
graficar_arbol(grafico) 


A = np.matrix([[0.7413249211356467, 0.25867507886435326, 0.7717717717717718, 0.707641196013289, 0.292358803986711, 0.2282282282282282, 0.744927536231884, 0.7370242214532872],
               [0.9794952681388013, 0.02050473186119872, 0.975975975975976, 0.9833887043189369, 0.01661129568106312, 0.024024024024024038, 0.9848484848484849, 0.9736842105263158],
              [0.9605678233438486, 0.039432176656151396, 0.963963963963964, 0.9568106312292359, 0.04318936877076407, 0.036036036036036, 0.9610778443113772, 0.96],
              [0.944794952681388, 0.05520504731861198, 0.948948948948949, 0.9401993355481728, 0.05980066445182719, 0.05105105105105101, 0.9461077844311377, 0.9433333333333334],
              [0.9006309148264984, 0.09936908517350163, 0.9519519519519519, 0.8438538205980066, 0.15614617940199338, 0.048048048048048075, 0.8708791208791209, 0.9407407407407408],
              [0.9668769716088328, 0.03312302839116721, 0.960960960960961, 0.973421926910299, 0.02657807308970095, 0.03903903903903905, 0.975609756097561, 0.9575163398692811],
              [0.944794952681388, 0.05520504731861198, 0.948948948948949, 0.9401993355481728, 0.05980066445182719, 0.05105105105105101, 0.9461077844311377, 0.9433333333333334],
              [0.916403785488959, 0.08359621451104104, 0.987987987987988, 0.8372093023255814, 0.16279069767441856, 0.012012012012011963, 0.8703703703703703, 0.984375]])
mi_df = pd.DataFrame(A)
nombres_filas = ["K-vecinos todas variables","K-vecinos 6 variables","Arbol de Decision (default)","Arbol de Decision (criterion='entropy',max_depth=2)","Arbol de Decision (criterion='gini', splitter='random', max_depth=3)","Arbol de Decision 6 predictoras (default)","Arbol de Decision 6 predictoras (criterion='entropy',max_depth=2)","Arbol de Decision 6 predictoras (criterion='gini', splitter='random', max_depth=3)"]
nombres_columnas = ["Precisi´on Global","Error Global","Precisi´on Positiva (PP)", "Precisi´on Negativa (PN)", "Falsos Positivos (FP)", "Falsos Negativos (FN)", "Asertividad Positiva (AP)", "Asertividad Negativa (AN)"]
mi_df = pd.DataFrame(A, index = nombres_filas, columns = nombres_columnas )
mi_df



#Ejercicio 3: [25 puntos] Esta pregunta utiliza los datos (tumores.csv). 

#1. Use el m´etodo de Arboles de Decisi´on en ´ Python para generar un modelo predictivo
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

instancia_arbol = DecisionTreeClassifier(random_state=0)

instancia_arbol.fit(X_train,y_train)

print("Las predicciones en Testing son: {}".format(instancia_arbol.predict(X_test)))


#2. Usando la funci´on programada en el ejercicio 1 de la tarea anterior, los datos tumores.csv

prediccion = instancia_arbol.predict(X_test)
MC = confusion_matrix(y_test, prediccion)    
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))



A = np.matrix([[0.8981723237597912, 0.10182767624020883, 0.9742857142857143, 0.09090909090909091, 0.9090909090909091, 0.02571428571428569, 0.9191374663072777, 0.25],
               [0.9843342036553525, 0.015665796344647487, 0.9914285714285714, 0.9090909090909091, 0.09090909090909094, 0.008571428571428563, 0.9914285714285714, 0.9090909090909091]])
mi_df = pd.DataFrame(A)
nombres_filas = ["K-vecinos", "Arbol de Decision"]
nombres_columnas = ["Precisi´on Global","Error Global","Precisi´on Positiva (PP)", "Precisi´on Negativa (PN)", "Falsos Positivos (FP)", "Falsos Negativos (FN)", "Asertividad Positiva (AP)", "Asertividad Negativa (AN)"]
mi_df = pd.DataFrame(A, index = nombres_filas, columns = nombres_columnas )
mi_df

#3. Grafique el ´arbol generado e interprete al menos dos reglas que se puedan extraer del
datos_plotear = export_graphviz(instancia_arbol, out_file=None,class_names=["No tumor", "Tumor"],
                feature_names=list(X.columns.values), filled=True)
grafico = graphviz.Source(datos_plotear) 
plt.rcParams['figure.figsize'] = [15, 15] # Tamaño del gráfico
graficar_arbol(grafico)  

##Podar
instancia_arbol2 = DecisionTreeClassifier(min_samples_leaf=150)
instancia_arbol2.fit(X_train,y_train)
prediccion = instancia_arbol2.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


datos_plotear = export_graphviz(instancia_arbol2, out_file=None,class_names=["No tumor", "Tumor"],
                feature_names=list(X.columns.values), filled=True)
grafico = graphviz.Source(datos_plotear) 
plt.rcParams['figure.figsize'] = [15, 15] # Tamaño del gráfico
graficar_arbol(grafico)


#Pregunta 4: [25 puntos] En este ejercicio vamos a predecir n´umeros escritos a mano (Hand
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


instancia_arbol = DecisionTreeClassifier(random_state=0)
instancia_arbol.fit(X,y)
print("Las predicciones en Testing son: {}".format(instancia_arbol.predict(X2)))


##2. Con la tabla de testing calcule la matriz de confusi´on, la precisi´on global, el error global
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

prediccion = instancia_arbol.predict(X2)
MC = confusion_matrix(y2, prediccion)
indices = indices_general_reducido(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


#3. Repita los ejercicios 1, 2 y 3 pero usando solamente los 3s, 5s y los 8s. ¿Mejora la predicci´on?
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

instancia_arbol = DecisionTreeClassifier(random_state=0)
instancia_arbol.fit(X,y)
print("Las predicciones en Testing son: {}".format(instancia_arbol.predict(X2)))

prediccion = instancia_arbol.predict(X2)
MC = confusion_matrix(y2, prediccion)
indices = indices_general_reducido(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))












