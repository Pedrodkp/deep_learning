import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.utils import np_utils

#obj: criar um deep autoencoder
#     fazemos todo o processamento, e na saida teremos algo parecido com a entrada
#     vamos comecar a reduzir com 784, reduzir para 128, depois 64, depois 32
#     depois vamos voltar para 64, 128 e entao 784

(previsores_treinamento, _), (previsores_teste, _) = mnist.load_data()
previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255

previsores_treinamento = previsores_treinamento.reshape(len(previsores_treinamento), np.prod(previsores_treinamento.shape[1:]))
previsores_teste = previsores_teste.reshape(len(previsores_teste), np.prod(previsores_teste.shape[1:]))

autoencoder = Sequential()

# Encode
autoencoder.add(Dense(units=128, activation='relu', input_dim=784))
autoencoder.add(Dense(units=64, activation='relu'))
autoencoder.add(Dense(units=32, activation='relu'))

# Decode
autoencoder.add(Dense(units=64, activation='relu'))
autoencoder.add(Dense(units=128, activation='relu'))
autoencoder.add(Dense(units=784, activation='sigmoid'))

# Usar
autoencoder.summary() #Total params: 222,384
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#como quero fazer a previsao da propria entrada, em vez de colocar a classe, coloco o proprio treinamento
#o mesmo, deveriam ser muito mais epochs
autoencoder.fit(previsores_treinamento, previsores_treinamento, epochs=50, batch_size=256, 
                validation_data=(previsores_teste, previsores_teste))

# Captura classificador
# Precisamos pegar as camadas antes do 32 (camada do meio) e incluir uma dentro da outra
dimensao_original = Input(shape=784,)
camada_encoder1 = autoencoder.layers[0]
camada_encoder2 = autoencoder.layers[1]
camada_encoder3 = autoencoder.layers[2]
encoder = Model(dimensao_original, camada_encoder3(camada_encoder2(camada_encoder1(dimensao_original))))
encoder.summary()
# Total params: 110,816 (aqui e onde o encoder serial salvo, a variavel encoder em si)

imagens_codificadas = encoder.predict(previsores_teste)
imagens_decodificadas = autoencoder.predict(previsores_teste)

# Ver
#perceba que a nitidez melhora
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
    plt.imshow(imagens_codificadas[indice_imagem].reshape(8,4))
    plt.xticks()
    plt.yticks()
    
    #imagem recontruida
    eixo = plt.subplot(10,10,i+1+numero_imagens*2)
    plt.imshow(imagens_decodificadas[indice_imagem].reshape(28,28))
    plt.xticks()
    plt.yticks()
