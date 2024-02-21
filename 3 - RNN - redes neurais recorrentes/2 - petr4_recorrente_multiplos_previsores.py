from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Antes trabalhamos somente com Open, agora a ideia e utilzar todos os atributos para a previsao do preco

base = pd.read_csv('petr4_treinamento.csv')
base = base.dropna()
base_treinamento = base.iloc[:,1:7].values

normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

previsores = []
preco_real = []
for i in range(90, 1242):
    #agora nao vamos mais de ZERO apenas, mais de 0:6 para pegar todos os atributos
    previsores.append(base_treinamento_normalizada[i-90:i, 0:6])
    preco_real.append(base_treinamento_normalizada[i, 0])
previsores, preco_real = np.array(previsores), np.array(preco_real)
#esse reshape nao e necessario, antes de executar e possivel ver que previsores tem o formato (1152,90,6)
#sendo 6 a quantidade de atributos, mas vou deixar a linha aqui para ter a referencia de como converter
#tambem nao ha problema manter, ele ira apenas fazer o reshape para o mesmo formato
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 6))

regressor = Sequential()
#no input shape tbm incrementamos de 1 para 6, por serem 6 atributos
regressor.add(LSTM(units=100,return_sequences=True, input_shape = (previsores.shape[1], 6)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

#tbm pode ser a linear, mas como fizemos a normalizacao de valores entre ZERO/UM
regressor.add(Dense(units=1, activation='sigmoid'))

#tbm vamos fazer um teste com o adam, mas o rmsprop se comporta bem
regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

##### CALLBACKS PARA MULTIPLOS PREVISORES #####

#callbacks nao sao diferentes de qlqr outro lugar, servem para coletar dados durante o processo,
#ou interagir com o treinamento durante sua execucao
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
#EarlyStopping (abortar se estiver muito ruim)
#monitor = funcao a ser monitorada
#min_delta = se nao melhorar na proxima exeucao o valor solicitado, para o aprendizado
#patiente = quantas epochs aceita sem melhora aceita
es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=True)
#ReduceLROnPlateau (ajuste fino na taxa de aprendizagem de nao melhora), atrs de controle sao os mesmo da EarlyStopping
#factor = atributo proprio para dizer qual o fator de reducao de aprendizagem
rlr = ReduceLROnPlateau(monitor='loss', patiente=5, factor=0.2, verbose=True)
#ModelCheckpoint salva os pesos durante as epochs, assim se algum resultado no meio do processo for bom, 
#pode ser usado em vez de apenas o ultimo
mcp = ModelCheckpoint(filepath='pesos.h5', monitor='loss', verbose=True, save_best_only=True)

regressor.fit(previsores, preco_real, epochs=100, batch_size=32, callbacks=[es, rlr, mcp])

##### COMPARAR COM TESTE #####
base_teste = pd.read_csv('petr4_teste.csv')
preco_real_teste = base_teste.iloc[:, 1:2].values

frames = [base, base_teste]
base_completa = pd.concat(frames)
base_completa = base_completa.drop('Date', axis=1)

entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
#nao faz o reshape como antes por estar usando uma dimensao maior (mais attr)
entradas = normalizador.transform(entradas)

X_teste = []
for i in range(90, 112):
    #antes ZERO agora 0:6 por serem mais attrs
    X_teste.append(entradas[i-90:i, 0:6])
X_teste = np.array(X_teste)
#o reshape aqui tbm nao e necessario, vai apenas deixar no formato que ja esta, mas mantemos para lembrar da etapa
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 6))
previsoes = regressor.predict(X_teste)

#precisamos tratar o normalizador do formato (22,1) para (22,6)
normalizador_previsao = MinMaxScaler(feature_range=(0,1))
#pegamos apenas o atributo que estamos prevendo
normalizador_previsao.fit_transform(base_treinamento[:,0:1])
previsoes = normalizador_previsao.inverse_transform(previsoes)

##### ANALISE #####

previsoes.mean()
preco_real_teste.mean()

plt.plot(preco_real_teste, color='red', label='Preço real')
plt.plot(previsoes, color='blue', label='Previsoes')
plt.title('Previsao preço das açoes')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()
    