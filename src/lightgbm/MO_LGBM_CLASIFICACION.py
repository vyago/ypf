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

system("clear") # Limpio entorno




#CARGO DATASETS
os.chdir("C:/Users/vyago/Desktop/Yago/Competencia/ypf")  # Directorio actual
train = pd.read_csv("../Dataset/dataset_train.csv")
test = pd.read_csv("../Dataset/dataset_test.csv")


#-------------------------------------------------------------

train = train.select_dtypes("number")
train["evento"] = train["delta_WHP"].apply( lambda x: 1 if x!=0 else 0)  # Genero feature evento de interferencia



y_train = train["evento"]
x_train = train[train.columns.drop(["delta_WHP","ID_FILA","evento"])]

X_train, X_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42)


train_data = lgb.Dataset(X_train, label=y_train)  


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
y_pred = modelo.predict(X_val)

fpr, tpr, threshold = metrics.roc_curve(y_val, y_pred)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()








