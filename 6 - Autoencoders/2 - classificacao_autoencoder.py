import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.utils import np_utils

#obj: como um algoritimo de classificacao trabalha com o autoencoder, por isso teremos a classe treinamento
#     vamos fazer duas redes dendas, uma usando a base codificada e outra nao para comparar

(previsores_treinamento, classe_treinamento), (previsores_teste, classe_teste) = mnist.load_data()
previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255
classe_dummy_treinamento = np_utils.to_categorical(classe_treinamento) #precisamos transformar a classe em colunas
classe_dummy_teste = np_utils.to_categorical(classe_teste)

previsores_treinamento = previsores_treinamento.reshape(len(previsores_treinamento), np.prod(previsores_treinamento.shape[1:]))
previsores_teste = previsores_teste.reshape(len(previsores_teste), np.prod(previsores_teste.shape[1:]))

autoencoder = Sequential()
#usamos 32 pixels, mas podemos usar mais para maior precisao
autoencoder.add(Dense(units=32, activation='relu', input_dim= 784)) #por padrao geralmente se utiliza a relu para este tipo de rede
autoencoder.add(Dense(units=784, activation='sigmoid')) #tem que ser a mesma quantidade da entrada na saida
autoencoder.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.fit(previsores_treinamento, previsores_treinamento, epochs=50, batch_size=256, validation_data=(previsores_teste, previsores_teste))

dimensao_original = Input(shape=(784,))
camada_encoder = autoencoder.layers[0]
encoder = Model(inputs=dimensao_original,outputs=camada_encoder(dimensao_original))

##### CLASSIFICACAO #####

previsores_treinamento_codificados = encoder.predict(previsores_treinamento)
previsores_teste_codificados = encoder.predict(previsores_teste)

### SEM REDUCAO ###
c1 = Sequential()
#units (784 entr + 10 saida)/2 = 397, essa formula e um ponto de partida, nada obrigatorio
#10 na saida por ser numeros de ZERO a NOVE
c1.add(Dense(units=397, activation='relu', input_dim=784))
c1.add(Dense(units=397, activation='relu'))
c1.add(Dense(units=10, activation='softmax'))
c1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
c1.fit(previsores_treinamento, classe_dummy_treinamento, batch_size=256, epochs=100, validation_data=(previsores_teste, classe_dummy_teste))
#val_accuracy: 0.9850 e levou quase 15 minutos

### COM REDUCAO ###
c2 = Sequential()
#units (32 entr + 10 saida)/2 = 21, essa formula e um ponto de partida, nada obrigatorio
#10 na saida por ser numeros de ZERO a NOVE
c2.add(Dense(units=21, activation='relu', input_dim=32))
c2.add(Dense(units=21, activation='relu'))
c2.add(Dense(units=10, activation='softmax'))
c2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
c2.fit(previsores_treinamento_codificados, classe_dummy_treinamento, batch_size=256, epochs=100, validation_data=(previsores_teste_codificados, classe_dummy_teste))
#val_accuracy: 0.9590 e levou menos de 1 minuto

#o ideial seria usar 5000 epochs, e para mais precisao e possivel tbm aumentar os pixel