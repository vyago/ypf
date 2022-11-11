import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from os import system
from category_encoders import TargetEncoder




#INPUT
input = 'HT004-FE002'
fe = input.split("-")[1]
#EXPERIMENTO
experimento = 'EN-009'

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

#-------------------------------------------------------------

x_train = x_train.select_dtypes("number")
x_train["evento"] = x_train["delta_WHP"].apply( lambda x: 1 if x!=0 else 0)  # Genero feature evento de interferencia



y_train = x_train["evento"]
x_train = x_train[x_train.columns.drop(["delta_WHP","ID_FILA","evento"])]
x_test = x_test[x_train.columns]
x_test = x_test.astype("float32")



train_data = lgb.Dataset(x_train, label=y_train)  


# Parámetros 

param = {'max_bin':128,
         'objective': 'binary',
         'learning_rate':0.19,
         'num_iterations':97,
         'min_data_in_leaf':1080,
         'feature_fraction':0.7169,
         'num_leaves':975}

param['metric'] = 'binary_logloss' # métrica


modelo=lgb.train(param,train_data)
y_pred = modelo.predict(x_test)
probas = pd.DataFrame({'probs':y_pred})

y_pred_bin = probas['probs'].apply(lambda x: 1 if x>0.7 else 0)
prediccion=pd.DataFrame()

prediccion_2 = pd.read_csv('C:/Users/vyago/Desktop/Yago/Competencia/Exp/EN-008/prediccion_2.csv',names=['id','pred'])

prediccion['ID_FILA'] =test['ID_FILA']
prediccion = pd.concat([prediccion,y_pred_bin,prediccion_2],axis=1)

prediccion['final_pred'] = prediccion['probs']*prediccion['pred']

if not os.path.isdir(f'../Exp/{experimento}'):
    os.makedirs(f'../Exp/{experimento}')


prediccion[['ID_FILA','final_pred']].to_csv(f"../Exp/{experimento}/prediccion_clasificador_reg.csv",index=False,header = False,sep=',')








