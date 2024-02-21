from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
No exemplo sobre a previsão dos valores das ações, nós trabalhamos com uma série temporal que apresenta os valores dia a dia. 
Porém, também podemos utilizar outros formatos de data, como por exemplo: horas, minutos ou segundos (dependendo do contexto). 
A base de dados desta tarefa possui essas características, na qual temos em cada registro o ano, o mês, o dia e a hora juntamente 
com o valor de poluição naquele momento e algumas características climáticas. Na imagem abaixo você pode visualizar alguns registros

O atributo No  é somente a contagem de registros (como uma chave primária), o year , month , day  e hour  indicam a dimensão temporal 
(de hora em hora); o atributo pm2.5  diz respeito ao nível de poluição (que faremos a previsão) e por fim, 
todos os outros atributos serão os previsores. Baseado nos atributos previsores, a ideia é indicar o nível de poluição 
em uma determinada hora.

Siga as seguintes dicas para essa atividade:

Use a função dropna()  para excluir valores faltantes
Os atributos No , year , month , day , hour  e cbwd  devem ser excluídos, pois em uma série temporal essas informações não 
são importantes (o cbwd  é somente um campo string)
Este é um problema com uma única saída (pm2.5 ) e múltiplos previsores
Você pode testar com vários valores de intervalos de tempo (começando com 10, por exemplo)
Depois de treinar a rede neural, gere o gráfico para visualizar os resultados das previsões

Os resultados das previsões ficaram próximos dos valores de poluição real?
'''

base = pd.read_csv('EXERCICIO 2 - poluicao/poluicao_treinamento.csv')
base = base.dropna()
base = base.drop('No', axis=1)
base = base.drop('year', axis=1)
base = base.drop('month', axis=1)
base = base.drop('day', axis=1)
base = base.drop('hour', axis=1)
base = base.drop('cbwd', axis=1)
attrs = 7

base_treinamento = base.iloc[:,0:attrs].values

normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

previsores = []
preco_real = []
for i in range(90, len(base_treinamento_normalizada)):
    previsores.append(base_treinamento_normalizada[i-90:i, 0:attrs])
    preco_real.append(base_treinamento_normalizada[i, 0])
previsores, preco_real = np.array(previsores), np.array(preco_real)
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], attrs))

regressor = Sequential()

regressor.add(LSTM(units=100,return_sequences=True, input_shape = (previsores.shape[1], attrs)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units=1, activation='linear'))

regressor.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

##### CALLBACKS PARA MULTIPLOS PREVISORES #####

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=True)
rlr = ReduceLROnPlateau(monitor='loss', patiente=5, factor=0.2, verbose=True)
mcp = ModelCheckpoint(filepath='pesos.h5', monitor='loss', verbose=True, save_best_only=True)

regressor.fit(previsores, preco_real, epochs=100, batch_size=32, callbacks=[es, rlr, mcp])

##### COMPARAR COM TESTE #####
base_teste = pd.read_csv('EXERCICIO 2 - poluicao/poluicao_teste.csv')
base_teste = base_teste.dropna()
base_teste = base_teste.drop('No', axis=1)
base_teste = base_teste.drop('year', axis=1)
base_teste = base_teste.drop('month', axis=1)
base_teste = base_teste.drop('day', axis=1)
base_teste = base_teste.drop('hour', axis=1)
base_teste = base_teste.drop('cbwd', axis=1)

pm25_teste = base_teste.iloc[:, 0:1].values

frames = [base, base_teste]
base_completa = pd.concat(frames)

entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
entradas = normalizador.transform(entradas)

X_teste = []
for i in range(90, len(base_teste)):
    X_teste.append(entradas[i-90:i, 0:attrs])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], attrs))
previsoes = regressor.predict(X_teste)

normalizador_previsao = MinMaxScaler(feature_range=(0,1))
normalizador_previsao.fit_transform(base_treinamento[:,0:1])
previsoes = normalizador_previsao.inverse_transform(previsoes)

##### ANALISE #####

previsoes.mean()
pm25_teste.mean()

plt.plot(pm25_teste, color='red', label='PM2.5 real')
plt.plot(previsoes, color='blue', label='Previsoes')
plt.title('Previsao da poluiçao na China')
plt.xlabel('Tempo')
plt.ylabel('PM2.5')
plt.legend()
plt.show()
    
    