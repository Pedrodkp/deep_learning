import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

#a ideia aqui e depois de treinar a rede, testar um novo registro

previsores = pd.read_csv('homer bart/entradas_personagens.csv')
classe = pd.read_csv('homer bart/saidas_personagens.csv')

classificador = Sequential()

#input_dim=6, because we have 6 columns in entry
classificador.add(Dense(units= 6, activation= 'relu', kernel_initializer= 'normal', input_dim = 6))
    
classificador.add(Dropout(0.2))

classificador.add(Dense(units= 6, activation= 'relu', kernel_initializer= 'normal'))
    
classificador.add(Dropout(0.2))

classificador.add(Dense(units= 1, activation='sigmoid'))

classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['binary_accuracy'])

classificador.fit(previsores, classe, batch_size=10, epochs=100)

#teste novo registro, como faria com a base teste
novo = np.array([[15.80, 8.34, 118, 50, 0.10, 0.26]])
previsao = classificador.predict(novo)
