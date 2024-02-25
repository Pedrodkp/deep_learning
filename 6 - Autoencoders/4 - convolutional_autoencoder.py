import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
#UpSampling2D (volta ao estado original) e quase o oposto do MaxPooling2D
from keras.utils import np_utils

#obj: criar um convolutional autoencoder

(previsores_treinamento, _), (previsores_teste, _) = mnist.load_data()
previsores_treinamento = previsores_treinamento.reshape((len(previsores_treinamento), 28, 28, 1))
previsores_teste = previsores_teste.reshape((len(previsores_teste), 28, 28, 1))

previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255

autoencoder = Sequential()

### Encoder

#relembrando, aplicar a convolucao aplica filtros para ter varias versoes dela com dimensao reduzida
#             depois o MaxPooling2D para mais uma reducao com as caracteristicas e depois o Flatten para transformar em vetor 
#             aplicando relu para matar os pixels negativos
autoencoder.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
autoencoder.add(MaxPooling2D(pool_size=(2,2)))

#padding=como a imagem sera passada
autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu',padding='same'))
autoencoder.add(MaxPooling2D(pool_size=(2,2),padding='same'))

#strides=indica de quantos em quantos pixels a imagem deve andar
autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu',padding='same', strides=(2,2)))

### Meio
#vai gerar um vetor 4 * 4 * 8 = 128
#exatamente o meio do processo
autoencoder.add(Flatten())
#volta para o formato original para iniciar o decoder
autoencoder.add(Reshape((4,4,8)))

### Decoder
autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu',padding='same'))
autoencoder.add(UpSampling2D(size=(2,2)))

autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu',padding='same'))
autoencoder.add(UpSampling2D(size=(2,2)))

# bem comum acontecer o erro de shape no autoencoder.fit la embaixo
#pois a imagem reconstruida esta com 32x32x1, mas a base e 28x28x1, logo como a primeira camada, essa nao deve ter o padding=same
autoencoder.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
autoencoder.add(UpSampling2D(size=(2,2)))

#resposta de fato, ou seja, a imagem decodifica
#aqui pode ter o padding=same, pois vai estar com o 28x28x1 original
autoencoder.add(Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid',padding='same'))

### Treinar
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#colocamos epochs=10 apenas para o exercicio, mas esse valor chega a ser ridiculo, deveria ser 5000, apenas para ir rapido (pode levar dias)
#100 epochs ja teriamos mais resultado que os valores anteriores
autoencoder.fit(previsores_treinamento, previsores_treinamento, epochs=10, batch_size=256, validation_data=(previsores_teste, previsores_teste))

### Extracao
#Vamos trabalhar de fato so com o flatten, para saber o nome do layer pegar do autoencoder.summary()
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('flatten').output)
encoder.summary()

### Usando
imagens_codificadas = encoder.predict(previsores_teste)
imagens_decodificadas = autoencoder.predict(previsores_teste)

### Ver
numero_imagens = 10
imagens_teste = np.random.randint(previsores_teste.shape[0], size=numero_imagens)
plt.figure(figsize=(18,18))
for i, indice_imagem in enumerate(imagens_teste):
    print(i)
    print(indice_imagem)
    
    #imagens originais
    eixo = plt.subplot(10,10,i+1)
    plt.imshow(previsores_teste[indice_imagem].reshape(28,28))
    plt.xticks()
    plt.yticks()
    
    #imagens codificadas
    eixo = plt.subplot(10,10,i+1+numero_imagens)
    plt.imshow(imagens_codificadas[indice_imagem].reshape(16,8))
    plt.xticks()
    plt.yticks()
    
    #imagem recontruida
    eixo = plt.subplot(10,10,i+1+numero_imagens*2)
    plt.imshow(imagens_decodificadas[indice_imagem].reshape(28,28))
    plt.xticks()
    plt.yticks()