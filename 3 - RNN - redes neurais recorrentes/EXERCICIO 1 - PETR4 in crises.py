from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import pandas as pd

#a ideia e prever o preco da acao passando por 2018 onde houve a crise e o preco caiu pela metade

base = pd.read_csv('EXERCICIO 1 - PETR4 in crises/petr4_treinamento_ex.csv')
base = base.dropna()

base_treinamento = base.iloc[:, 1:2].values

from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

previsores = []
preco_real = []
for i in range(90, len(base_treinamento_normalizada)):
    previsores.append(base_treinamento_normalizada[i-90:i, 0]) #coloca os 90 anteriores
    preco_real.append(base_treinamento_normalizada[i, 0]) #coloca o registro atual

previsores, preco_real = np.array(previsores), np.array(preco_real)
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

from keras.layers import LSTM

regressor = Sequential()
regressor.add(LSTM(units=100,return_sequences=True, input_shape = (previsores.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units=1, activation='sigmoid'))

regressor.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

regressor.fit(previsores, preco_real, epochs=100, batch_size= 32)

##### EFETIVAMENTE PREVER #####

base_teste = pd.read_csv('EXERCICIO 1 - PETR4 in crises/petr4_teste_ex.csv')
preco_real_teste = base_teste.iloc[:, 1:2].values

base_completa = pd.concat((base['Open'], base_teste['Open']), axis=0)
entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

X_teste = []
for i in range(90, 90+len(base_teste)):
    X_teste.append(entradas[i-90:i, 0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))
previsoes = regressor.predict(X_teste)

previsoes = normalizador.inverse_transform(previsoes)

previsoes.mean()
preco_real_teste.mean()

##### GRAFICO #####
import matplotlib.pyplot as plt

plt.plot(preco_real_teste, color='red', label='Preço real')
plt.plot(previsoes, color='blue', label='Previsoes')
plt.title('Previsao preço das açoes')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legent()
plt.show()
    