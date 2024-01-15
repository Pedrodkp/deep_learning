import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
#tive que mudar para lib nova
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

#objetivo, aplicar o tunning, normal esta 0.9533333333333334, melhorar

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
classe = label_encoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criar_rede(optimizer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units=neurons, activation=activation, input_dim=4))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=neurons, activation=activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=3, activation='softmax'))
    
    classificador.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics='accuracy')
    return classificador

classificador = KerasClassifier(build_fn=criar_rede)
parametros = {'batch_size' : [10, 30],
              'epochs' : [500, 1000],
              'model__optimizer' : ['adam', 'sgd'],
              'model__activation' : ['relu', 'tanh'],
              'model__neurons' : [4, 8, 16]
              }

grid_search = GridSearchCV(estimator = classificador, param_grid = parametros, scoring= 'accuracy', cv=10)

grid_search = grid_search.fit(previsores, classe)
melhor_parametros = grid_search.best_params_
#{'batch_size': 10, 'epochs': 500, 'model__activation': 'relu', 'model__neurons': 8, 'model__optimizer': 'adam'}
melhor_precisao = grid_search.best_score_ #0.9866666666666667
