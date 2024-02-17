import tensorflow as tf
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

##EXERCICIO, conseguir media maior que 90

def criar_rede():
    classificador = Sequential()
    
    classificador.add(Dense(units= 8, activation= 'relu', \
                            kernel_initializer= 'normal', input_dim = 30))
        
    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units= 8, activation= 'relu', \
                            kernel_initializer= 'normal'))
        
    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units= 1, activation='sigmoid'))
        
    classificador.compile(optimizer='adam', loss='binary_crossentropy', \
                          metrics= ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn=criar_rede, epochs=100, batch_size=10)

resultados = cross_val_score(estimator=classificador, X=previsores, y=classe, cv=10, scoring='accuracy')

media = resultados.mean()
