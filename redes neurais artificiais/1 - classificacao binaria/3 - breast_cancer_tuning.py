import tensorflow as tf
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

#a ideia aqui e que o keras faça um tuning de parametros, ex:
#qtd de neuronios, camadas, tipos de funcoes de passo
#demora horas, para ele faz o ajuste fino do treinamento sozinho

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criar_rede(optimizer, loos, kernel_initializer, activation, neurons):
    classificador = Sequential()
    #posso variar qualquer camada
    classificador.add(Dense(units= neurons, activation= activation, \
                            kernel_initializer= kernel_initializer, input_dim = 30))
        
    classificador.add(Dropout(0.2))
    #posso deixar uma camada fixa
    classificador.add(Dense(units= 16, activation= 'relu', \
                            kernel_initializer= 'random_uniform'))
        
    classificador.add(Dropout(0.2))
    #a ultima nao faz muito sentido, ja que queremos apenas uma resposta no final: maligno/benigno
    classificador.add(Dense(units= 1, activation='sigmoid'))
    
    classificador.compile(optimizer=optimizer, loss=loos, \
                          metrics= ['binary_accuracy'])
    return classificador

#epochs podem ser 500, 1000, 10000, 1milhao para sistemas comerciais
#neurons tbm e interessante testar com o dobro do padrao, ex: 32
classificador = KerasClassifier(build_fn=criar_rede)
parametros = {'batch_size' : [10, 30],
              'epochs' : [50, 100],
              'model__optimizer' : ['adam', 'sgd'],
              'model__loos' : ['binary_crossentropy', 'hinge'],
              'model__kernel_initializer' : ['random_uniform', 'normal'],
              'model__activation' : ['relu', 'tanh'],
              'model__neurons' : [16, 8]
              }

#cv, o normal e 10, feito com 5 por performance no aprendizado
grid_search = GridSearchCV(estimator = classificador, param_grid = parametros, scoring= 'accuracy', cv=5)

grid_search = grid_search.fit(previsores, classe)
melhor_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_