import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.utils import np_utils

#obj: implementar um deep autoencoder com cifar20

(previsores_treinamento, _), (previsores_teste, _) = cifar10.load_data()
previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255

previsores_treinamento = previsores_treinamento.reshape(len(previsores_treinamento), np.prod(previsores_treinamento.shape[1:]))
previsores_teste = previsores_teste.reshape(len(previsores_teste), np.prod(previsores_teste.shape[1:]))

autoencoder = Sequential()

# Encode
#32 * 32 * 3 = 3072
#3072 / 2 = 1536
autoencoder.add(Dense(units=1536, activation='relu', input_dim=3072))
autoencoder.add(Dense(units=768, activation='relu')) #1536/2=768
autoencoder.add(Dense(units=384, activation='relu')) #768/2=384

# Decode
autoencoder.add(Dense(units=768, activation='relu'))
autoencoder.add(Dense(units=1536, activation='relu'))
autoencoder.add(Dense(units=3072, activation='sigmoid'))

# Usar
autoencoder.summary() #Total params: 11,803,392
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.fit(previsores_treinamento, previsores_treinamento, epochs=100, batch_size=256, 
                validation_data=(previsores_teste, previsores_teste))

# Captura classificador
dimensao_original = Input(shape=3072,)
camada_encoder1 = autoencoder.layers[0]
camada_encoder2 = autoencoder.layers[1]
camada_encoder3 = autoencoder.layers[2]
encoder = Model(dimensao_original, camada_encoder3(camada_encoder2(camada_encoder1(dimensao_original))))
encoder.summary() # Total params: 

imagens_codificadas = encoder.predict(previsores_teste)
imagens_decodificadas = autoencoder.predict(previsores_teste)

# Ver
numero_imagens = 10
imagens_teste = np.random.randint(previsores_teste.shape[0], size=numero_imagens)
    #aqui temos que tentar formar um quadrado para enxergar, achando a RAIZ do shape, √384 = 19,595917942
    #nao podemos arredondar e precisam ser numeros inteiros
    #logo nem sempre e possivel, podemos entao tentar uma multiplicacao que de exatamente 384 para tentar enxergar
    #uma combinacao pode ser 16x24 = 384, podemos inverter, etc, enfim, o que facilitar enxergar, isso para cada reshape
plt.figure(figsize=(18,18))
for i, indice_imagem in enumerate(imagens_teste):
    print(i)
    print(indice_imagem)
    
    #imagens originais
    eixo = plt.subplot(10,10,i+1)
    #48*64=3072
    plt.imshow(previsores_teste[indice_imagem].reshape(32,32,3))
    plt.xticks()
    plt.yticks()
    
    #imagens codificadas
    eixo = plt.subplot(10,10,i+1+numero_imagens)
    plt.imshow(imagens_codificadas[indice_imagem].reshape(16,24))
    plt.xticks()
    plt.yticks()
    
    #imagem recontruida
    eixo = plt.subplot(10,10,i+1+numero_imagens*2)
    plt.imshow(imagens_decodificadas[indice_imagem].reshape(32,32,3))
    plt.xticks()
    plt.yticks()
