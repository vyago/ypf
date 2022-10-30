import pandas as pd
import numpy as np

import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
from hyperopt import hp


import os
import lightgbm as lgb

nombre_experimento="HT_102_clasificacion"
#CARGO DATASET
os.chdir("C:/Users/vyago/Desktop/Yago/Competencia/ypf")  # Directorio actual
train = pd.read_csv("../Dataset/dataset_train.csv")
train["evento"] = train["delta_WHP"].apply( lambda x: 1 if x!=0 else 0)  # Genero feature evento de interferencia


train = train.select_dtypes("number")
y_train = train["evento"]
X_train = train[train.columns.drop(["delta_WHP","ID_FILA"])]#,"ID_EVENTO"])]

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
   params["max_bin"] = 128
   params['objective'] = 'binary'
   params['feature_pre_filter'] = False
   # realiza n_folds de cross validation
   cv_results = lgb.cv(params, train_set, num_boost_round = 9999999,
                       nfold = n_folds, early_stopping_rounds = int(50+5/params["learning_rate"]) , 
                       metrics = 'binary_logloss', seed = 50,stratified=False, return_cvbooster= True)
   
   num_iteration = cv_results["cvbooster"].best_iteration
   
   run_time = timer() - start
   
   # Extrae el mejor score
   best_score = np.max(cv_results['binary_logloss-mean'])
   
   # El loss se debe minimizar
   loss = best_score
   
   # Impulsando las iteraciones que arrojaron el mayor score en CV
   n_estimators = int(np.argmax(cv_results['binary_logloss-mean']) + 1)
   
   # Escribe sobre el archivo CSV ('a' significa append)
   of_connection = open(out_file, 'a')
   writer = csv.writer(of_connection)
   writer.writerow([loss, params,num_iteration, ITERATION, n_estimators, 
                   run_time])
   
   # Dictionary con informacion para la evaluación
   return {'loss': loss, 'params': params, 'numero_de_iteracion':num_iteration, 'iteration': ITERATION,
           'estimators': n_estimators, 'train_time': run_time, 
           'status': STATUS_OK}
   
space = {   
'num_leaves': hp.quniform('num_leaves', 100, 2000, 1),
'learning_rate': hp.loguniform('learning_rate', np.log(0.01),np.log(0.2)),
'min_data_in_leaf': hp.quniform('min_data_in_leaf', 200, 4000, 5),
'feature_fraction': hp.uniform('feature_fraction', 0.1, 1.0)
}
   
   # Algoritmo de optimización
tpe_algorithm = tpe.suggest

# Lleva el registro de los resultados
bayes_trials = Trials()

# archivo para guardar los primeros resultados
out_file = f'../Exp/{nombre_experimento}.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# escribe la cabecera de los archivos
writer.writerow(['loss', 'params','numero_de_iteracion' 'iteration', 'estimators', 'train_time'])
of_connection.close()


# Variable Global
global  ITERATION
ITERATION = 0
MAX_EVALS = 100

# Crea un dataset lgb
train_set = lgb.Dataset(X_train, label = y_train)

# Corre la optimización
best = fmin(fn = objective, space = space, algo = tpe.suggest,
            max_evals = MAX_EVALS, trials = bayes_trials, 
            rstate=np.random.default_rng(50))

