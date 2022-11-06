import pandas as pd
import numpy as np

import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
from hyperopt import hp

from sklearn.model_selection import train_test_split

from category_encoders import TargetEncoder



import os
import lightgbm as lgb




#FEATURE ENGINEERING QUE HAY QUE OPTIMIZAR
input='FE002'

#COLOCO NOMBRE DEL EXPERIMENTO
experimento=f"HT004-{input}"


#Experimento aplicando codificación de target a las variables categóricas. Le sumo interacción entre las mismas
# Debido a esto, no se podrá optimizar a traves de cross validation, solo podré hacer 1 validación.




#CARGO DATASET

os.chdir("C:/Users/vyago/Desktop/Yago/Competencia/ypf")  # Directorio actual
train = pd.read_csv(f"../Exp/{input}/train.csv")  # Cargo Dataset de entrenamiento
X = train[train.columns.drop(["delta_WHP","ID_FILA"])] # Me quedó con todas las columnas salvo el target
y = np.sqrt(train["delta_WHP"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

enc = TargetEncoder(cols=['PAD_HIJO','HIJO','PADRE'], min_samples_leaf=20, smoothing=10).fit(X_train, y_train)

X_train= enc.transform(X_train)
X_test = enc.transform(X_test)
X_train = X_train.select_dtypes("number")
X_test = X_test[X_train.columns]

# INTERACCIÓN ENTRE VARIABLES CATEGÓRICAS 

var_cat = ['PAD_HIJO','HIJO','PADRE']

for feature_1 in var_cat:
    for feature_2 in var_cat:
        if feature_1!=feature_2:
            X_train[f'{feature_1}-{feature_2}'] = X_train[f'{feature_1}']*X_train[f'{feature_2}']  #CREO VARIABLES DE INTERACCIONES EN TRAIN
            X_test[f'{feature_1}-{feature_2}'] = X_test[f'{feature_1}']*X_test[f'{feature_2}']  #CREO VARIABLES DE INTERACCIONES EN TEST





MAX_EVALS = 500
N_FOLDS = 5

def objective(params, n_folds = N_FOLDS):
   """Función objetivo para la Optimización de hiperparametros del Gradient Boosting Machine"""
   
   # Llevar el conteo de iteraciones
   global ITERATION
   ITERATION += 1
   
   # Recupera el subsample si se encuentra, en caso contrario se asigna 1.0
   #subsample = params['boosting_type'].get('subsample', 1.0)
   
   # Extrae el boosting type
   params['boosting_type'] = 'gbdt'
   params['subsample'] = 1.0
  
   # Se asegura que los parametros que tienen que ser enteros sean enteros
   for parameter_name in ['num_leaves', 
                          'min_data_in_leaf']:
       params[parameter_name] = int(params[parameter_name])
   start = timer()
   params["max_bin"] = 256
   params['objective'] = 'regression'
   params['feature_pre_filter'] = False
   params['early_stopping_rounds'] = int(50+10/params["learning_rate"])
   params['metrics'] = 'rmse'
   # realiza n_folds de cross validation
   
   """
   cv_results = lgb.cv(params, train_set, num_boost_round = 9999999,
                       nfold = n_folds, early_stopping_rounds = int(50+5/params["learning_rate"]) , 
                       metrics = 'rmse', seed = 50,stratified=False, return_cvbooster= True)
   """
   modelo = lgb.train(params=params,train_set=train_set,num_boost_round= 999999, valid_sets=test_set)
   
   num_iteration = modelo.best_iteration
   
   run_time = timer() - start
   
   # Extrae el mejor score
   
   
   key = list(modelo.best_score) #clave del diccionario 
   best_score = modelo.best_score[key[0]]['rmse']
   
   # El loss se debe minimizar
   loss = best_score
   
      
   # Escribe sobre el archivo CSV ('a' significa append)
   of_connection = open(out_file, 'a')
   writer = csv.writer(of_connection)
   writer.writerow([loss, params,num_iteration, ITERATION, 
                   run_time,STATUS_OK])
   
   # Dictionary con informacion para la evaluación
   return {'loss': loss, 'params': params, 'numero_de_iteracion':num_iteration, 'iteration': ITERATION, 'train_time': run_time, 
           'status': STATUS_OK}
   
space = {   
'num_leaves': hp.quniform('num_leaves', 500, 1500, 1),
'learning_rate': hp.loguniform('learning_rate', np.log(0.01),np.log(0.2)),
'min_data_in_leaf': hp.quniform('min_data_in_leaf', 500, 1500, 5),
'feature_fraction': hp.uniform('feature_fraction', 0.2, 0.5)
}
   
   # Algoritmo de optimización
tpe_algorithm = tpe.suggest

# Lleva el registro de los resultados
bayes_trials = Trials()

# archivo para guardar los primeros resultados

if not os.path.isdir(f'../Exp/{experimento}'):
    os.makedirs(f'../Exp/{experimento}')

out_file = f'../Exp/{experimento}/HT.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# escribe la cabecera de los archivos
writer.writerow(['loss', 'params','numero_de_iteracion' 'iteration', 'estimators', 'train_time'])
of_connection.close()


# Variable Global
global  ITERATION
ITERATION = 0
MAX_EVALS = 20

# Crea un dataset lgb
train_set = lgb.Dataset(X_train, label = y_train)
test_set = lgb.Dataset(X_test,label = y_test)

# Corre la optimización
best = fmin(fn = objective, space = space, algo = tpe.suggest,
            max_evals = MAX_EVALS, trials = bayes_trials, 
            rstate=np.random.default_rng(50))

