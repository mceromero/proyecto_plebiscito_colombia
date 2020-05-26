# -*- coding: utf-8 -*-

# import modules
import pandas as pd

# joblib -> loading model
import joblib


# Utils
def return_prob(model, data):
    '''
    '''
    d = model.predict_proba(data)
    return [i[1] for i in d]



# load model
'''
Caragar en esta línea de código el modelo creado
en main.py
'''
root_model = './result/model.best.pkl'
model = joblib.load(root_model)

# load data
'''
Cargar base de datos completa para utilizar el modelo de predicción

Este script considera que la base de datos incluye el campo

- Content (Texto)

El cual será evaluado por el modelo para predecir sentimiento.
'''
main_root = ''
data = pd.read_csv(
	main_root, encoding='utf-8', low_memory=False
)

data['polarity_class'] = model.predict(data['Content'])
data['polarity_index'] = return_prob(model, data['Content'])

# saving results
data.to_csv('./results.csv', encoding='utf-8', index=False)
