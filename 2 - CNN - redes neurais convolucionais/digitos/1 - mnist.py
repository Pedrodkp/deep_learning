import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

opcional = True

(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()
#ver na aba Plots
plt.imshow(X_treinamento[0], cmap='gray')
plt.imshow(X_treinamento[2])
plt.imshow(X_treinamento[5])
plt.title('Classe '+str(y_treinamento[0]))
plt.title('Classe '+str(y_treinamento[2]))

#1 no fim e um canal, ou seja, escala de cinza e normaliza tamanho 28*28
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 28, 28, 1)
previsores_treinamento[0]
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)

#mudar para float, senao ao usar 0a1 vai perder precisao
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

#em vez de usar byte (256), usar uma escala de 0 a 1 por performace para RGB ou tom cinza
previsores_treinamento /= 255
previsores_teste /= 255
previsores_treinamento[0]

#criar variaveis do tipo dummy 0=000
#10 = n de classes, feito isso por estarmos tratando um probl de prrobabilidades de saida
#valor em y esta indo ate 9
classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

### ETAPA 1 - operador de convolucao
classificador = Sequential()
#filters=32 por que nao queremos que ele use os kernels padroes, queremos que ele invente alguns e teste
#   o ideal e comecar com 64 kernels e ir seguindo os multiplos
#kernel_size=3x3 por que as imagens sao pequenas, podemos aumentar para imagens maiores
#strides=como a janela se move, nao mudamos para usar o default (1,1) um para direita e um para baixo
classificador.add(Conv2D(32, (3,3), input_shape=(28, 28, 1), activation='relu'))

#OPCIONAL, da mesma maneira que normalizamos os dados transformando 255 de valor de 0a1, melhora velocidade
#podemos normalizar os dados que sera feito o pooling, ex: se sao valores de 1a5, serao reduzidos a valores de 0a1
if opcional:
    classificador.add(BatchNormalization())

### ETAPA 2 - pooling
#pool_size=2x2, tamanho da janela do pooling, variar de acordo com o tamanho da saida da etapa anterior
classificador.add(MaxPooling2D(pool_size=(2,2)))

#OPCIONAL, podemos colocar uma ou mais camadas de convoluçao completa, sem input_shape, melhora precisao
if opcional:
    classificador.add(Conv2D(32, (3,3), activation='relu'))
    classificador.add(BatchNormalization())
    classificador.add(MaxPooling2D(pool_size=(2,2)))

### ETAPA 3 - flattening
classificador.add(Flatten())

### ETAPA 4 - rede neural densa
#camada intermediaria
#estimar o tamanho do mapa de caracteristicas, mas o normal e ir em multiplos binarios 128, 256, 512 e etc
classificador.add(Dense(units=128, activation='relu'))

#OPCIONAL dropout por ter muitos valores de entrada, e mais uma camada oculta
if opcional:
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=128, activation='relu'))
    classificador.add(Dropout(0.2))

#camada de saida, numeros de 0 a 9
classificador.add(Dense(units=10, activation='softmax'))
classificador.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#epochs=natural que sejam muitas mais, mas demora muitas horas, 5 para testes
#validaton_data=ira validar os dados a cada epoca
classificador.fit(previsores_treinamento, classe_treinamento, batch_size=128, epochs=5, validation_data=(previsores_teste, classe_teste))

#observar o val_acc para ver o desempenho do treinamento, com 5 epocas chega a 0.98, mas com 100 a 200 epocas sera quase 100
#evaluate e o mesmo que o validation_data
resultado = classificador.evaluate(previsores_teste, classe_teste)
