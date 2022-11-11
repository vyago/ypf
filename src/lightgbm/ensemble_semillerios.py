import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import lightgbm as lgb

from category_encoders import TargetEncoder


#INPUT
input = 'HT004-FE002'
fe = input.split("-")[1]
#EXPERIMENTO
experimento = 'EN-008'

#CARGO DATASETS
os.chdir("C:/Users/vyago/Desktop/Yago/Competencia/ypf")  # Directorio actual
train = pd.read_csv(f"../Exp/{fe}/train.csv")
test = pd.read_csv(f"../Exp/{fe}/test.csv")
y = np.sqrt(np.absolute(train["delta_WHP"]))


enc = TargetEncoder(cols=['PAD_HIJO','HIJO','PADRE'], min_samples_leaf=20, smoothing=10).fit(train, y)

x_train= enc.transform(train)
x_test = enc.transform(test)


# INTERACCIÓN ENTRE VARIABLES CATEGÓRICAS 

var_cat = ['PAD_HIJO','HIJO','PADRE']

for feature_1 in var_cat:
    for feature_2 in var_cat:
        if feature_1!=feature_2:
            x_train[f'{feature_1}-{feature_2}'] = x_train[f'{feature_1}']*x_train[f'{feature_2}']  #CREO VARIABLES DE INTERACCIONES EN TRAIN
            x_test[f'{feature_1}-{feature_2}'] = x_test[f'{feature_1}']*x_test[f'{feature_2}']  #CREO VARIABLES DE INTERACCIONES EN TEST


x_train = x_train.select_dtypes("number")

x_train = x_train[x_train.columns.drop(["delta_WHP","ID_FILA"])]

x_test = x_test[x_train.columns]
x_test = x_test.astype("float32")


train_data = lgb.Dataset(x_train, label=y)


# Parámetros 

params = {'feature_fraction': 0.31293008809690764, 'learning_rate': 0.07988247330134716, 'min_data_in_leaf': 610, 'num_leaves': 718, 'boosting_type': 'gbdt', 'subsample': 1.0, 'max_bin': 256, 'objective': 'regression', 'feature_pre_filter': False}
params['metric'] = 'rmse' # métrica
params["num_iterations"] = 7240
params["max_depth"] = -1


def generador_numeros(cantidad):
    semillas=[]
    for num in range(0,cantidad):
        semillas.append(1+num+1234*num)
    return semillas

semillas = generador_numeros(100)


predicciones = pd.DataFrame()

for semilla in semillas:
    
    params['seed'] = semilla
    modelo=lgb.train(params,train_data)
    predicciones[f'modelo_seed_{semilla}'] = np.square(modelo.predict(x_test))
    


predicciones["mean"] = predicciones.median(axis=1) 


prediccion=test[["ID_FILA"]]

prediccion = pd.concat([prediccion,predicciones["mean"]],axis=1)


if not os.path.isdir(f'../Exp/{experimento}'):
    os.makedirs(f'../Exp/{experimento}')

prediccion.to_csv(f"../Exp/{experimento}/prediccion.csv",header=False,index=False)
predicciones.to_csv(f"../Exp/{experimento}/predicciones.csv",index=False)

predicciones_2 = pd.read_csv('C:/Users/vyago/Desktop/Yago/Competencia/Exp/EN-008/predicciones.csv')
prediccion_2=pd.DataFrame()

predicciones_2 = predicciones_2[predicciones_2.columns.drop('mean')]
predicciones_2['mean'] = predicciones_2.mean(axis=1)
predicciones_2['mean'] = predicciones_2['mean'].apply(lambda x : 0 if x<0.01 else x)
prediccion_2['ID_FILA'] =test['ID_FILA']
prediccion_2 = pd.concat([prediccion_2,predicciones_2['mean']],axis=1)
prediccion_2.to_csv(f"../Exp/{experimento}/prediccion_3.csv",index=False,header=False)


