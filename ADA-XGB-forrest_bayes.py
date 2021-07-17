# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 08:53:19 2020

@author: rzamoram
"""

#Pregunta 1: [40 puntos] En este ejercicio usaremos los datos (voces.csv).
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import numpy  as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
from matplotlib import colors as mcolors
import pandas as pd


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


def indices_general(MC, nombres = None):
    precision_global = np.sum(MC.diagonal()) / np.sum(MC)
    error_global = 1 - precision_global
    precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
    if nombres!=None:
        precision_categoria.columns = nombres
    return {"Matriz de Confusión":MC, 
            "Precisión Global":precision_global, 
            "Error Global":error_global, 
            "Precisión por categoría":precision_categoria}
    
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

#2. Use Bosques Aleatorios, ADABoosting y XGBoosting 
X = datos.iloc[:,:20] 
print(X.head())
y = datos.iloc[:,20:21] 
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)


##Bosques
instancia_bosque = RandomForestClassifier(n_estimators=10, random_state=0)

instancia_bosque.fit(X_train,y_train.iloc[:,0].values)

print("Las predicciones en Testing son: {}".format(instancia_bosque.predict(X_test)))

prediccion = instancia_bosque.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))

importancia = instancia_bosque.feature_importances_
print(importancia)
etiquetas = X_train.columns.values
y_pos = np.arange(len(etiquetas))
plt.barh(y_pos, importancia, align='center', alpha=0.5)
plt.yticks(y_pos, etiquetas)


###XGBoosting
instancia_potenciacion = GradientBoostingClassifier(n_estimators=10, random_state=0)
instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)
print("Las predicciones en Testing son: {}".format(instancia_potenciacion.predict(X_test)))

prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))
    
importancia = instancia_potenciacion.feature_importances_
print(importancia)
etiquetas = X_train.columns.values
y_pos = np.arange(len(etiquetas))
plt.barh(y_pos, importancia, align='center', alpha=0.5)
plt.yticks(y_pos, etiquetas)


###ADA Boosting
instancia_potenciacion = AdaBoostClassifier(n_estimators=10, random_state=0)
instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)
print("Las predicciones en Testing son: {}".format(instancia_potenciacion.predict(X_test)))

prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))
    
importancia = instancia_potenciacion.feature_importances_
print(importancia)
etiquetas = X_train.columns.values
y_pos = np.arange(len(etiquetas))
plt.barh(y_pos, importancia, align='center', alpha=0.5)
plt.yticks(y_pos, etiquetas)
#3. Usando la funci´on programada en el ejercicio 1 de la tarea anterior

def indices_general_extra(MC, nombres = None):
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

##Bosques
instancia_bosque = RandomForestClassifier(n_estimators=10, random_state=0)
instancia_bosque.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_bosque.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_extra(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


###XGBoosting
instancia_potenciacion = GradientBoostingClassifier(n_estimators=10, random_state=0)
instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_extra(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


###ADA Boosting
instancia_potenciacion = AdaBoostClassifier(n_estimators=10, random_state=0)
instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_extra(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))



A = np.matrix([[0.7413249211356467, 0.25867507886435326, 0.7717717717717718, 0.707641196013289, 0.292358803986711, 0.2282282282282282, 0.744927536231884, 0.7370242214532872],
               [0.9794952681388013, 0.02050473186119872, 0.975975975975976, 0.9833887043189369, 0.01661129568106312, 0.024024024024024038, 0.9848484848484849, 0.9736842105263158],
              [0.9605678233438486, 0.039432176656151396, 0.963963963963964, 0.9568106312292359, 0.04318936877076407, 0.036036036036036, 0.9610778443113772, 0.96],
              [0.944794952681388, 0.05520504731861198, 0.948948948948949, 0.9401993355481728, 0.05980066445182719, 0.05105105105105101, 0.9461077844311377, 0.9433333333333334],
              [0.9006309148264984, 0.09936908517350163, 0.9519519519519519, 0.8438538205980066, 0.15614617940199338, 0.048048048048048075, 0.8708791208791209, 0.9407407407407408],
              [0.9668769716088328, 0.03312302839116721, 0.960960960960961, 0.973421926910299, 0.02657807308970095, 0.03903903903903905, 0.975609756097561, 0.9575163398692811],
              [0.944794952681388, 0.05520504731861198, 0.948948948948949, 0.9401993355481728, 0.05980066445182719, 0.05105105105105101, 0.9461077844311377, 0.9433333333333334],
              [0.916403785488959, 0.08359621451104104, 0.987987987987988, 0.8372093023255814, 0.16279069767441856, 0.012012012012011963, 0.8703703703703703, 0.984375],
              [0.9810725552050473, 0.018927444794952675, 0.975975975975976, 0.9867109634551495, 0.013289036544850474, 0.024024024024024038, 0.9878419452887538, 0.9737704918032787],
              [0.9779179810725552, 0.02208201892744477, 0.9819819819819819, 0.973421926910299, 0.02657807308970095, 0.018018018018018056, 0.9761194029850746, 0.979933110367893],
              [0.9652996845425867, 0.034700315457413256, 0.987987987987988, 0.9401993355481728, 0.05980066445182719, 0.012012012012011963, 0.9481268011527377, 0.9860627177700348]])
mi_df = pd.DataFrame(A)
nombres_filas = ["K-vecinos todas variables","K-vecinos 6 variables","Arbol de Decision (default)","Arbol de Decision (criterion='entropy',max_depth=2)","Arbol de Decision (criterion='gini', splitter='random', max_depth=3)","Arbol de Decision 6 predictoras (default)","Arbol de Decision 6 predictoras (criterion='entropy',max_depth=2)","Arbol de Decision 6 predictoras (criterion='gini', splitter='random', max_depth=3)", "Bosques", "XGBoosting", "ADA Boosting"]
nombres_columnas = ["Precisi´on Global","Error Global","Precisi´on Positiva (PP)", "Precisi´on Negativa (PN)", "Falsos Positivos (FP)", "Falsos Negativos (FN)", "Asertividad Positiva (AP)", "Asertividad Negativa (AN)"]
mi_df = pd.DataFrame(A, index = nombres_filas, columns = nombres_columnas )
mi_df



#4. Repita los ejercicios 1-3, pero esta vez use una combinaci´on diferente de los par´ametros
##Bosques
instancia_bosque = RandomForestClassifier(n_estimators=7, max_depth=2, random_state=0)
instancia_bosque.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_bosque.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_extra(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


###XGBoosting
instancia_potenciacion = GradientBoostingClassifier(n_estimators=7, max_depth=2, random_state=0)
instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_extra(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


###ADA Boosting
instancia_potenciacion = AdaBoostClassifier(n_estimators=7, random_state=0)
instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_extra(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))



A = np.matrix([[0.7413249211356467, 0.25867507886435326, 0.7717717717717718, 0.707641196013289, 0.292358803986711, 0.2282282282282282, 0.744927536231884, 0.7370242214532872],
               [0.9794952681388013, 0.02050473186119872, 0.975975975975976, 0.9833887043189369, 0.01661129568106312, 0.024024024024024038, 0.9848484848484849, 0.9736842105263158],
              [0.9605678233438486, 0.039432176656151396, 0.963963963963964, 0.9568106312292359, 0.04318936877076407, 0.036036036036036, 0.9610778443113772, 0.96],
              [0.944794952681388, 0.05520504731861198, 0.948948948948949, 0.9401993355481728, 0.05980066445182719, 0.05105105105105101, 0.9461077844311377, 0.9433333333333334],
              [0.9006309148264984, 0.09936908517350163, 0.9519519519519519, 0.8438538205980066, 0.15614617940199338, 0.048048048048048075, 0.8708791208791209, 0.9407407407407408],
              [0.9668769716088328, 0.03312302839116721, 0.960960960960961, 0.973421926910299, 0.02657807308970095, 0.03903903903903905, 0.975609756097561, 0.9575163398692811],
              [0.944794952681388, 0.05520504731861198, 0.948948948948949, 0.9401993355481728, 0.05980066445182719, 0.05105105105105101, 0.9461077844311377, 0.9433333333333334],
              [0.916403785488959, 0.08359621451104104, 0.987987987987988, 0.8372093023255814, 0.16279069767441856, 0.012012012012011963, 0.8703703703703703, 0.984375],
              [0.9810725552050473, 0.018927444794952675, 0.975975975975976, 0.9867109634551495, 0.013289036544850474, 0.024024024024024038, 0.9878419452887538, 0.9737704918032787],
              [0.9779179810725552, 0.02208201892744477, 0.9819819819819819, 0.973421926910299, 0.02657807308970095, 0.018018018018018056, 0.9761194029850746, 0.979933110367893],
              [0.9652996845425867, 0.034700315457413256, 0.987987987987988, 0.9401993355481728, 0.05980066445182719, 0.012012012012011963, 0.9481268011527377, 0.9860627177700348],
              [0.9542586750788643, 0.045741324921135695, 0.975975975975976, 0.9302325581395349, 0.06976744186046513, 0.024024024024024038, 0.9393063583815029, 0.9722222222222222],
              [0.9621451104100947, 0.03785488958990535, 0.9429429429429429, 0.9833887043189369, 0.01661129568106312, 0.0570570570570571, 0.9843260188087775, 0.9396825396825397],
              [0.9589905362776026, 0.04100946372239744, 0.987987987987988, 0.9269102990033222, 0.07308970099667778, 0.012012012012011963, 0.9373219373219374, 0.9858657243816255]])
mi_df = pd.DataFrame(A)
nombres_filas = ["K-vecinos todas variables","K-vecinos 6 variables","Arbol de Decision (default)","Arbol de Decision (criterion='entropy',max_depth=2)","Arbol de Decision (criterion='gini', splitter='random', max_depth=3)",
                 "Arbol de Decision 6 predictoras (default)","Arbol de Decision 6 predictoras (criterion='entropy',max_depth=2)","Arbol de Decision 6 predictoras (criterion='gini', splitter='random', max_depth=3)", 
                 "Bosques", "XGBoosting", "ADA Boosting", "Bosques (n_estimators=7, max_depth=2, random_state=0) ", "XGBoosting(n_estimators=7, max_depth=2, random_state=0)", "ADA Boosting (n_estimators=7, random_state=0)"]
nombres_columnas = ["Precisi´on Global","Error Global","Precisi´on Positiva (PP)", "Precisi´on Negativa (PN)", "Falsos Positivos (FP)", "Falsos Negativos (FN)", "Asertividad Positiva (AP)", "Asertividad Negativa (AN)"]
mi_df = pd.DataFrame(A, index = nombres_filas, columns = nombres_columnas )
mi_df



#5. Repita los ejercicios 1-4, pero esta vez use 2 combinaciones diferentes de selecci´on de 6
X = datos.iloc[:,[1,3,5,8,11,12]] 
print(X.head())
y = datos.iloc[:,20:21]
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)


##Bosques
instancia_bosque = RandomForestClassifier(n_estimators=10, random_state=0)
instancia_bosque.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_bosque.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_extra(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))



###XGBoosting
instancia_potenciacion = GradientBoostingClassifier(n_estimators=10, random_state=0)
instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_extra(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))
    

###ADA Boosting
instancia_potenciacion = AdaBoostClassifier(n_estimators=10, random_state=0)
instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_extra(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))
    


###Diferentes parametros
##Bosques
instancia_bosque = RandomForestClassifier(n_estimators=700, max_depth=200, random_state=0)
instancia_bosque.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_bosque.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_extra(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


###XGBoosting
instancia_potenciacion = GradientBoostingClassifier(n_estimators=700, max_depth=200, random_state=0)
instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_extra(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


###ADA Boosting
instancia_potenciacion = AdaBoostClassifier(n_estimators=700, random_state=0)
instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_extra(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))

##############


X = datos.iloc[:,[1,5,9,12,13,17]] 
print(X.head())
y = datos.iloc[:,20:21]
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)


##Bosques
instancia_bosque = RandomForestClassifier(n_estimators=10, random_state=0)
instancia_bosque.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_bosque.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_extra(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))



###XGBoosting
instancia_potenciacion = GradientBoostingClassifier(n_estimators=10, random_state=0)
instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_extra(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))
    

###ADA Boosting
instancia_potenciacion = AdaBoostClassifier(n_estimators=10, random_state=0)
instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_extra(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))
    


###Diferentes parametros
##Bosques
instancia_bosque = RandomForestClassifier(n_estimators=700, max_depth=200, random_state=0)
instancia_bosque.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_bosque.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_extra(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


###XGBoosting
instancia_potenciacion = GradientBoostingClassifier(n_estimators=700, max_depth=200, random_state=0)
instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_extra(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


###ADA Boosting
instancia_potenciacion = AdaBoostClassifier(n_estimators=700, random_state=0)
instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_extra(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))



A = np.matrix([[0.7413249211356467, 0.25867507886435326, 0.7717717717717718, 0.707641196013289, 0.292358803986711, 0.2282282282282282, 0.744927536231884, 0.7370242214532872],
               [0.9794952681388013, 0.02050473186119872, 0.975975975975976, 0.9833887043189369, 0.01661129568106312, 0.024024024024024038, 0.9848484848484849, 0.9736842105263158],
              [0.9605678233438486, 0.039432176656151396, 0.963963963963964, 0.9568106312292359, 0.04318936877076407, 0.036036036036036, 0.9610778443113772, 0.96],
              [0.944794952681388, 0.05520504731861198, 0.948948948948949, 0.9401993355481728, 0.05980066445182719, 0.05105105105105101, 0.9461077844311377, 0.9433333333333334],
              [0.9006309148264984, 0.09936908517350163, 0.9519519519519519, 0.8438538205980066, 0.15614617940199338, 0.048048048048048075, 0.8708791208791209, 0.9407407407407408],
              [0.9668769716088328, 0.03312302839116721, 0.960960960960961, 0.973421926910299, 0.02657807308970095, 0.03903903903903905, 0.975609756097561, 0.9575163398692811],
              [0.944794952681388, 0.05520504731861198, 0.948948948948949, 0.9401993355481728, 0.05980066445182719, 0.05105105105105101, 0.9461077844311377, 0.9433333333333334],
              [0.916403785488959, 0.08359621451104104, 0.987987987987988, 0.8372093023255814, 0.16279069767441856, 0.012012012012011963, 0.8703703703703703, 0.984375],
              [0.9810725552050473, 0.018927444794952675, 0.975975975975976, 0.9867109634551495, 0.013289036544850474, 0.024024024024024038, 0.9878419452887538, 0.9737704918032787],
              [0.9779179810725552, 0.02208201892744477, 0.9819819819819819, 0.973421926910299, 0.02657807308970095, 0.018018018018018056, 0.9761194029850746, 0.979933110367893],
              [0.9652996845425867, 0.034700315457413256, 0.987987987987988, 0.9401993355481728, 0.05980066445182719, 0.012012012012011963, 0.9481268011527377, 0.9860627177700348],
              [0.9542586750788643, 0.045741324921135695, 0.975975975975976, 0.9302325581395349, 0.06976744186046513, 0.024024024024024038, 0.9393063583815029, 0.9722222222222222],
              [0.9621451104100947, 0.03785488958990535, 0.9429429429429429, 0.9833887043189369, 0.01661129568106312, 0.0570570570570571, 0.9843260188087775, 0.9396825396825397],
              [0.9589905362776026, 0.04100946372239744, 0.987987987987988, 0.9269102990033222, 0.07308970099667778, 0.012012012012011963, 0.9373219373219374, 0.9858657243816255],
              [0.9810725552050473, 0.018927444794952675, 0.975975975975976, 0.9867109634551495, 0.013289036544850474, 0.024024024024024038, 0.9878419452887538, 0.9737704918032787],
              [0.9684542586750788, 0.03154574132492116, 0.975975975975976, 0.9601328903654485, 0.039867109634551534, 0.024024024024024038, 0.9643916913946587, 0.9730639730639731],
              [0.9574132492113565, 0.04258675078864349, 0.9669669669669669, 0.946843853820598, 0.05315614617940201, 0.033033033033033066, 0.9526627218934911, 0.9628378378378378],
              [0.9826498422712934, 0.017350157728706628, 0.9819819819819819, 0.9833887043189369, 0.01661129568106312, 0.018018018018018056, 0.9849397590361446, 0.9801324503311258],
              [0.9652996845425867, 0.034700315457413256, 0.96996996996997, 0.9601328903654485, 0.039867109634551534, 0.03003003003003002, 0.9641791044776119, 0.9665551839464883],
              [0.9637223974763407, 0.0362776025236593, 0.9819819819819819, 0.9435215946843853, 0.056478405315614655, 0.018018018018018056, 0.9505813953488372, 0.9793103448275862],
              [0.9779179810725552, 0.02208201892744477, 0.978978978978979, 0.9767441860465116, 0.023255813953488413, 0.02102102102102099, 0.978978978978979, 0.9767441860465116],
              [0.973186119873817, 0.02681388012618302, 0.972972972972973, 0.973421926910299, 0.02657807308970095, 0.027027027027026973, 0.9759036144578314, 0.9701986754966887],
              [0.9652996845425867, 0.034700315457413256, 0.987987987987988, 0.9401993355481728, 0.05980066445182719, 0.012012012012011963, 0.9481268011527377, 0.9860627177700348],
              [0.9779179810725552, 0.02208201892744477, 0.975975975975976, 0.9800664451827242, 0.019933554817275767, 0.024024024024024038, 0.9818731117824774, 0.9735973597359736],
              [0.9652996845425867, 0.034700315457413256, 0.963963963963964, 0.9667774086378738, 0.03322259136212624, 0.036036036036036, 0.9697885196374623, 0.9603960396039604],
              [0.9716088328075709, 0.028391167192429068, 0.975975975975976, 0.9667774086378738, 0.03322259136212624, 0.024024024024024038, 0.9701492537313433, 0.9732441471571907]])
mi_df = pd.DataFrame(A)
nombres_filas = ["K-vecinos todas variables","K-vecinos 6 variables","Arbol de Decision (default)","Arbol de Decision (criterion='entropy',max_depth=2)","Arbol de Decision (criterion='gini', splitter='random', max_depth=3)",
                 "Arbol de Decision 6 predictoras (default)","Arbol de Decision 6 predictoras (criterion='entropy',max_depth=2)","Arbol de Decision 6 predictoras (criterion='gini', splitter='random', max_depth=3)", 
                 "Bosques", "XGBoosting", "ADA Boosting", "Bosques (n_estimators=7, max_depth=2, random_state=0) ", "XGBoosting(n_estimators=7, max_depth=2, random_state=0)", "ADA Boosting (n_estimators=7, random_state=0)",
                 "Bosques 6 predictoras (opcion 1)", "XGBoosting 6 predictoras (opcion 1)", "ADA Boosting 6 predictoras(opcion 1)",
                 "Bosques 6 predictoras (opcion 1)(n_estimators=700, max_depth=200, random_state=0)", "XGBoosting 6 predictoras (opcion 1)(n_estimators=700, max_depth=200, random_state=0)", "ADA Boosting 6 predictoras(opcion 1)(n_estimators=700, random_state=0)",
                 "Bosques 6 predictoras (opcion 2)", "XGBoosting 6 predictoras (opcion 2)", "ADA Boosting 6 predictoras(opcion 2)",
                 "Bosques 6 predictoras (opcion 2)(n_estimators=700, max_depth=200, random_state=0)", "XGBoosting 6 predictoras (opcion 2)(n_estimators=700, max_depth=200, random_state=0)", "ADA Boosting 6 predictoras(opcion 2)(n_estimators=700, random_state=0)"]
nombres_columnas = ["Precisi´on Global","Error Global","Precisi´on Positiva (PP)", "Precisi´on Negativa (PN)", "Falsos Positivos (FP)", "Falsos Negativos (FN)", "Asertividad Positiva (AP)", "Asertividad Negativa (AN)"]
mi_df = pd.DataFrame(A, index = nombres_filas, columns = nombres_columnas )
mi_df


#Ejercicio 2: [25 puntos] Esta pregunta utiliza los datos (tumores.csv). 

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


##Bosques
instancia_bosque = RandomForestClassifier(n_estimators=10, random_state=0)

instancia_bosque.fit(X_train,y_train.iloc[:,0].values)

print("Las predicciones en Testing son: {}".format(instancia_bosque.predict(X_test)))

prediccion = instancia_bosque.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))

importancia = instancia_bosque.feature_importances_
print(importancia)
etiquetas = X_train.columns.values
y_pos = np.arange(len(etiquetas))
plt.barh(y_pos, importancia, align='center', alpha=0.5)
plt.yticks(y_pos, etiquetas)


###XGBoosting
instancia_potenciacion = GradientBoostingClassifier(n_estimators=10, random_state=0)
instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)
print("Las predicciones en Testing son: {}".format(instancia_potenciacion.predict(X_test)))

prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))
    
importancia = instancia_potenciacion.feature_importances_
print(importancia)
etiquetas = X_train.columns.values
y_pos = np.arange(len(etiquetas))
plt.barh(y_pos, importancia, align='center', alpha=0.5)
plt.yticks(y_pos, etiquetas)


###ADA Boosting
instancia_potenciacion = AdaBoostClassifier(n_estimators=10, random_state=0)
instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)
print("Las predicciones en Testing son: {}".format(instancia_potenciacion.predict(X_test)))

prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))
    
importancia = instancia_potenciacion.feature_importances_
print(importancia)
etiquetas = X_train.columns.values
y_pos = np.arange(len(etiquetas))
plt.barh(y_pos, importancia, align='center', alpha=0.5)
plt.yticks(y_pos, etiquetas)



#2. Usando la funci´on programada en el ejercicio 1 de la tarea anterior


##Bosques
instancia_bosque = RandomForestClassifier(n_estimators=10, random_state=0)
instancia_bosque.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_bosque.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_extra(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


###XGBoosting
instancia_potenciacion = GradientBoostingClassifier(n_estimators=10, random_state=0)
instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_extra(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


###ADA Boosting
instancia_potenciacion = AdaBoostClassifier(n_estimators=10, random_state=0)
instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general_extra(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))
    
    
A = np.matrix([[0.8981723237597912, 0.10182767624020883, 0.9742857142857143, 0.09090909090909091, 0.9090909090909091, 0.02571428571428569, 0.9191374663072777, 0.25],
               [0.9843342036553525, 0.015665796344647487, 0.9914285714285714, 0.9090909090909091, 0.09090909090909094, 0.008571428571428563, 0.9914285714285714, 0.9090909090909091],
               [0.9817232375979112, 0.018276762402088753, 0.9885714285714285, 0.9090909090909091, 0.09090909090909094, 0.011428571428571455, 0.9914040114613181, 0.8823529411764706],
               [0.9765013054830287, 0.023498694516971286, 0.9942857142857143, 0.7878787878787878, 0.21212121212121215, 0.005714285714285672, 0.9802816901408451, 0.9285714285714286],
               [0.9843342036553525, 0.015665796344647487, 0.9914285714285714, 0.9090909090909091, 0.09090909090909094, 0.008571428571428563, 0.9914285714285714, 0.9090909090909091],
               [0.9921671018276762, 0.007832898172323799, 0.9914285714285714, 1.0, 0.0, 0.008571428571428563, 1.0, 0.9166666666666666],
               [0.9843342036553525, 0.015665796344647487, 0.9914285714285714, 0.9090909090909091, 0.09090909090909094, 0.008571428571428563, 0.9914285714285714, 0.9090909090909091],
               [0.9869451697127938, 0.01305483028720622, 0.9914285714285714, 0.9393939393939394, 0.06060606060606055, 0.008571428571428563, 0.994269340974212, 0.9117647058823529]])
mi_df = pd.DataFrame(A)
nombres_filas = ["K-vecinos", "Arbol de Decision", "Bosques aleatorios", "XGBoosting", "ADA Boosting", "Bosques aleatorios (n_estimators=1000, max_depth=250, random_state=0)", "XGBoosting (n_estimators=1000, max_depth=250, random_state=0)", "ADA Boosting (n_estimators=1000, random_state=0)"]
nombres_columnas = ["Precisi´on Global","Error Global","Precisi´on Positiva (PP)", "Precisi´on Negativa (PN)", "Falsos Positivos (FP)", "Falsos Negativos (FN)", "Asertividad Positiva (AP)", "Asertividad Negativa (AN)"]
mi_df = pd.DataFrame(A, index = nombres_filas, columns = nombres_columnas )
mi_df


#Pregunta 3: [30 puntos] La idea de este ejercicio es programar una Clase en Python

x = np.matrix([[1,0,0], [1,0,1], [1,1,0], [1,1,1]])
y = np.matrix([[1], [1], [1], [0]])
x
y


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
import scipy.stats as sps
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

# Load in the data and define the column labels

pasada = os.getcwd()
os.chdir("C:/Users/rzamoram/Documents/Machine Learning/Métodos Supervisados con Python/Clase 01")
print(os.getcwd())
dataset = pd.read_csv('voces.csv',delimiter=',',decimal=".")
print(dataset.shape)
print(dataset.head())
print(dataset.info())



for label in dataset.columns:
    dataset[label] = LabelEncoder().fit(dataset[label]).transform(dataset[label])

    
    
Tree_model = DecisionTreeClassifier(criterion="entropy",max_depth=1)

X = dataset.drop('genero',axis=1)
Y = dataset['genero'].where(dataset['genero']==1,-1)

predictions = np.mean(cross_validate(Tree_model,X,Y,cv=100)['test_score'])


print('The accuracy is: ',predictions*100,'%')



class ConsensoPropio:
    def __init__(self,dataset,T,test_dataset):
        self.dataset = dataset
        self.T = T
        self.test_dataset = test_dataset
        self.alphas = None
        self.models = None
        self.accuracy = []
        self.predictions = None
    def fit(self):
        # parte descriptiva
        X = self.dataset.drop(['genero'],axis=1)
        Y = self.dataset['genero'].where(self.dataset['genero']==1,-1)
        # los pesos 
        evalua = pd.DataFrame(Y.copy())
        evalua['weights'] = 1/len(self.dataset) 
        # corre el algoritmo
        alphas = [] 
        models = []
        for t in range(self.T):
            
            ##Arbol de decision
            Tree_model = DecisionTreeClassifier(criterion="entropy",max_depth=1) 
            model = Tree_model.fit(X,Y,sample_weight=np.array(evalua['weights'])) 
            models.append(model)
            predictions = model.predict(X)
            score = model.score(X,Y)
            evalua['predictions'] = predictions
            evalua['evalua'] = np.where(evalua['predictions'] == evalua['target'],1,0)
            evalua['misclassified'] = np.where(evalua['predictions'] != evalua['target'],1,0)
            accuracy = sum(evalua['evalua'])/len(evalua['evalua'])
            misclassification = sum(evalua['misclassified'])/len(evalua['misclassified'])
            err = np.sum(evalua['weights']*evalua['misclassified'])/np.sum(evalua['weights'])
            alpha = np.log((1-err)/err)
            alphas.append(alpha)
            evalua['weights'] *= np.exp(alpha*evalua['misclassified'])
            
            ##Kneighbors
            kneighbor = KNeighborsClassifier(n_neighbors=3)
            model2 = kneighbor.fit(X,Y,sample_weight=np.array(evalua['weights'])) 
            model2.append(model2)
            predictions2 = model2.predict(X)
            score2 = model2.score(X,Y)
            evalua['predictions'] = predictions2
            evalua['evalua'] = np.where(evalua['predictions'] == evalua['target'],1,0)
            evalua['misclassified'] = np.where(evalua['predictions'] != evalua['target'],1,0)
            accuracy2 = sum(evalua['evalua'])/len(evalua['evalua'])
            misclassification2 = sum(evalua['misclassified'])/len(evalua['misclassified'])
            err2 = np.sum(evalua['weights']*evalua['misclassified'])/np.sum(evalua['weights'])
            alpha2 = np.log((1-err2)/err2)
            alphas2.append(alpha2)
            evalua['weights'] *= np.exp(alpha2*evalua['misclassified'])
            
            ###XGBoosting
            xgboosting = GradientBoostingClassifier(n_estimators=10, random_state=0)
            model3 = xgboosting.fit(X,Y,sample_weight=np.array(evalua['weights'])) 
            model3.append(model3)
            predictions3 = model3.predict(X)
            score3 = model3.score(X,Y)
            evalua['predictions'] = predictions3
            evalua['evalua'] = np.where(evalua['predictions'] == evalua['target'],1,0)
            evalua['misclassified'] = np.where(evalua['predictions'] != evalua['target'],1,0)
            accuracy3 = sum(evalua['evalua'])/len(evalua['evalua'])
            misclassification3 = sum(evalua['misclassified'])/len(evalua['misclassified'])
            err3 = np.sum(evalua['weights']*evalua['misclassified'])/np.sum(evalua['weights'])
            alpha3 = np.log((1-err3)/err3)
            alphas3.append(alpha3)
            evalua['weights'] *= np.exp(alpha3*evalua['misclassified'])      
        
            ###ADA Boosting
            ada = AdaBoostClassifier(n_estimators=10, random_state=0)
            model4 = ada.fit(X,Y,sample_weight=np.array(evalua['weights'])) 
            model4.append(model4)
            predictions4 = model4.predict(X)
            score4 = model4.score(X,Y)
            evalua['predictions'] = predictions4
            evalua['evalua'] = np.where(evalua['predictions'] == evalua['target'],1,0)
            evalua['misclassified'] = np.where(evalua['predictions'] != evalua['target'],1,0)
            accuracy4 = sum(evalua['evalua'])/len(evalua['evalua'])
            misclassification4 = sum(evalua['misclassified'])/len(evalua['misclassified'])
            err4 = np.sum(evalua['weights']*evalua['misclassified'])/np.sum(evalua['weights'])
            alpha4 = np.log((1-err4)/err4)
            alphas4.append(alpha4)
            evalua['weights'] *= np.exp(alpha4*evalua['misclassified'])
        
        
        self.alphas = alphas
        self.models = models
        
        self.alpha2 = alpha2
        self.models2 = model2
        
        self.alpha3 = alpha3
        self.models3 = model3
        
        self.alpha4 = alpha4
        self.models4 = model4
            
    def predict(self):
        X_test = self.test_dataset.drop(['genero'],axis=1).reindex(range(len(self.test_dataset)))
        Y_test = self.test_dataset['genero'].reindex(range(len(self.test_dataset))).where(self.dataset['genero']==1,-1)
        
        accuracy = []
        predictions = []
        
        for alpha,model in zip(self.alphas,self.models):
            prediction = alpha*model.predict(X_test) # We use the predict method for the single decisiontreeclassifier models in the list
            predictions.append(prediction)
            self.accuracy.append(np.sum(np.sign(np.sum(np.array(predictions),axis=0))==Y_test.values)/len(predictions[0]))
            
        self.predictions = np.sign(np.sum(np.array(predictions),axis=0))

   
        

        



number_of_base_learners = 50

fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)


for i in range(number_of_base_learners):
    model = ConsensoPropio(dataset,i,dataset)
    model.fit()
    model.predict()

       


















