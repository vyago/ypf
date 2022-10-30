import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import lightgbm as lgb


#CARGO DATASETS
os.chdir("C:/Users/vyago/Desktop/Yago/Competencia/ypf")  # Directorio actual
train = pd.read_csv("../Dataset/dataset_train.csv")
test = pd.read_csv("../Dataset/dataset_test.csv")

dataset = pd.concat([train,test],axis=0)

experimento = 'FE001'

# VARIABLES DUMMIES PARA FEATURES CATEGÓRICAS
oil_gas = pd.get_dummies(dataset["FLUIDO"])
campo = pd.get_dummies(dataset["CAMPO"])


# INTERACCIÓN ENTRE DISTANCIAS 

distancias = ['D3D', 'D2D', 'DZ', 'AZ']
interacciones = pd.DataFrame()

for feature_1 in distancias:
    for feature_2 in distancias:
        if feature_1!=feature_2:
            interacciones[f'{feature_1}-{feature_2}']=dataset[f'{feature_1}']*dataset[f'{feature_2}']/1000


dataset = pd.concat([dataset,interacciones,oil_gas,campo],axis=1)

#SEPARO EN TRAIN Y TEST

test = dataset[dataset["delta_WHP"].isna()]
train = dataset[dataset["delta_WHP"].notna()]


if not os.path.isdir(f'../Exp/{experimento}'):
    os.makedirs(f'../Exp/{experimento}')
 
train.to_csv(f"../Exp/{experimento}/train.csv",index=False)
test.to_csv(f"../Exp/{experimento}/test.csv",index=False)