from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv('petr4_treinamento.csv')
base = base.dropna()

#a ideia e prever mais de um valor de saida, Open e High
base_treinamento = base.iloc[:,1:2].values
base_valor_maximo = base.iloc[:,2:3].values

normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)
base_valor_maximo_normalizada = normalizador.fit_transform(base_valor_maximo)

previsores = []
preco_real1 = []
preco_real2 = []
for i in range(90, 1242):
    #vamos usar apenas o Open como previsor, mas vamos dar multiplas saidas
    #mas e possivel ter multiplos previsores com multiplas saidas
    previsores.append(base_treinamento_normalizada[i-90:i, 0])
    preco_real1.append(base_treinamento_normalizada[i, 0])
    preco_real2.append(base_valor_maximo_normalizada[i, 0])
previsores, preco_real1, preco_real2 = np.array(previsores), np.array(preco_real1), np.array(preco_real2)

#temos 1 atributo, entao temos que aumentar uma dimensao
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

#como temos 2 vetores, precisamos criar uma variavel que une ambos
preco_real = np.column_stack((preco_real1, preco_real2))

regressor = Sequential()
#apenas 1 atributo previsor, logo ultimo para e UM
regressor.add(LSTM(units=100,return_sequences=True, input_shape = (previsores.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

#agora temos 2 neuronios, para duas saidas
regressor.add(Dense(units=2, activation='linear'))

#tbm vamos fazer um teste com o adam, mas o rmsprop se comporta bem
regressor.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

regressor.fit(previsores, preco_real, epochs=100, batch_size=32)

##### COMPARAR COM TESTE #####
base_teste = pd.read_csv('petr4_teste.csv')
preco_real_open = base_teste.iloc[:, 1:2].values
preco_real_high = base_teste.iloc[:, 2:3].values

base_completa = pd.concat((base['Open'], base_teste['Open']), axis=0)
entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

X_teste = []
#90 records for the first date
#112 = 90+22 records for the test dataset
for i in range(90, 112):
    X_teste.append(entradas[i-90:i, 0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))

previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

##### ANALISE #####

previsoes.mean()
preco_real_open.mean()

plt.plot(preco_real_open, color='red', label='Preço abertura real')
plt.plot(preco_real_high, color='black', label='Preço alta real')
plt.plot(previsoes[:,0], color='blue', label='Previsao abertura')
plt.plot(previsoes[:,1], color='orange', label='Previsoes alta')
plt.title('Previsao preço das açoes')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()
    