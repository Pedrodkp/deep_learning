from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import pandas as pd

#a ideia e prever o preco de uma acao

base = pd.read_csv('petr4_treinamento.csv')
base = base.dropna()

#aqui escolhemos qual coluna queremos prever do modelo de treinamento, neste exemplo a Open mas poderia ser qualquer uma
base_treinamento = base.iloc[:, 1:2].values #nao considera upperbound, entao vai pegar apenas a coluna 1 (.values para gerar numpy arrary)

#normalizacao dos dados (para aumentar performance)
from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

#para trabalhar com series, precisamos transformar a diferenca de dias em atributos
#devemos comecar a montar series a partir da posicao que determina o tamanho da base para previsoes
#ex: se a base comeca em 1/1/24, e impossivel preceber 2/1/24, se vamos usar 90 dias, podemos comecar a montar series em 1/4/24
#    que sera o primeiro dia que podera ser "pervisto", basicamente pra eu prever o dia de amanha, tenho que ter 90 dias no passado
previsores = []
preco_real = []
#90 records for the first date
#1242 end of the training dataset
for i in range(90, 1242):
    previsores.append(base_treinamento_normalizada[i-90:i, 0]) #coloca os 90 anteriores
    preco_real.append(base_treinamento_normalizada[i, 0]) #coloca o registro atual

#numpy format
previsores, preco_real = np.array(previsores), np.array(preco_real)
#mudamos para 3 dimensoes para o keras utilizar
#previsores.shape[0] = 1152
#previsores.shape[1] = 90
#indicadores = 1 (por ser apenas uma coluna, poderiam ser quantas necessarias para quantos valores necessarios)
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

from keras.layers import LSTM

regressor = Sequential()
#units = celulas de memoria, precisa ser grande suficiente para capturar a variacao temporal
#return_sequences = True, ativado quando existe mais de uma camada
regressor.add(LSTM(units=100,return_sequences=True, input_shape = (previsores.shape[1], 1)))
regressor.add(Dropout(0.3))

#este tipo de rede neural precisa de muitas camadas para funcionar corretamente, vai testando os valores de units camada a camada
regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

#camada densa para resposta final
#activation = linar, para apenas passar o valor que chegou, por ser regressao
#* podemos tbm testar a sigmoid por que os dados estao normalizados
regressor.add(Dense(units=1, activation='linear'))

#loss = como o erro e tratado
#metrics = como o dado e visto
regressor.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

regressor.fit(previsores, preco_real, epochs=100, batch_size= 32)

##### EFETIVAMENTE PREVER #####

base_teste = pd.read_csv('petr4_teste.csv')
preco_real_teste = base_teste.iloc[:, 1:2].values

#precisamos ter os 90 precos anteriores tbm, por isso somamos a base original com a teste
base_completa = pd.concat((base['Open'], base_teste['Open']), axis=0)
entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
entradas = entradas.reshape(-1, 1)
#nao usamos o metodo fit_transform por que ele se adapta e ja foi feito, agora queremos um transform igual antes, logo so transform
entradas = normalizador.transform(entradas)

X_teste = []
#90 records for the first date
#112 = 90+22 records for the test dataset
for i in range(90, 112):
    X_teste.append(entradas[i-90:i, 0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))
previsoes = regressor.predict(X_teste)

#processo inverso da normalizacao para enxergar melhor
previsoes = normalizador.inverse_transform(previsoes)
### agora basta comparar previsoes X preco_real_teste no variable explorer do ide

#podemos tbm comparar as medias das comparacoes
previsoes.mean()
preco_real_teste.mean()

##### GRAFICO #####
import matplotlib.pyplot as plt

#need select all lines to see de graphs in Plots
plt.plot(preco_real_teste, color='red', label='Preço real')
plt.plot(previsoes, color='blue', label='Previsoes')
plt.title('Previsao preço das açoes')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legent()
plt.show()
    