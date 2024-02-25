import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense

 #perceba que nao pegamos a classe, pois vamos codificar e decodificar
(previsores_treinamento, _), (previsores_teste, _) = mnist.load_data()
previsores_treinamento = previsores_treinamento.astype('float32') / 255 #poderia ser o MinMaxScale, apenas para lembrar o que e feito
previsores_teste = previsores_teste.astype('float32') / 255

#precisamos converter a matriz em array, multiplicando o X por Y
#np.prod e multiplicacao, o shape[1:] e o 28x28, logo em vez de 28x28 teremos o 784
previsores_treinamento = previsores_treinamento.reshape(len(previsores_treinamento), np.prod(previsores_treinamento.shape[1:]))
previsores_teste = previsores_teste.reshape(len(previsores_teste), np.prod(previsores_teste.shape[1:]))

##### ENCODE/DECODE #####
#as camadas serao: entr 784, escondida 32, saida 784. Agora por que 32? Voce decide isso.
#na pratica a compactacao sera de 784 / 32 = 24,5
fator_compactacao = 784 / 32
autoencoder = Sequential()
#a primeira camada e o encode, a segunda o decode
autoencoder.add(Dense(units=32, activation='relu', input_dim= 784)) #por padrao geralmente se utiliza a relu para este tipo de rede
#vamos usar a 'sigmoid' por que fizemos a normalizacao entre 0/1, senao o interessante seria usar a tangente hiperbolica 'tanh'
autoencoder.add(Dense(units=784, activation='sigmoid')) #tem que ser a mesma quantidade da entrada na saida
#25120 = (784*32)+32 (baias na escondida)
#25872 = (784*32)+784 (saida)
#soma 50992 parametros
autoencoder.summary()
autoencoder.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
#compara entrada com entrada, nao entrada com classe
#autoencoders precisam de muitas epocas, 50 e apenas para ver executando
autoencoder.fit(previsores_treinamento, previsores_treinamento, epochs=50, batch_size=256, validation_data=(previsores_teste, previsores_teste))

##### VER #####
#na pratica esta fazendo o mesmo que acima
dimensao_original = Input(shape=(784,))
camada_encoder = autoencoder.layers[0]
encoder = Model(inputs=dimensao_original,outputs=camada_encoder(dimensao_original))
encoder.summary()

#ps, em um ambiente comercial basta salvar o autoencoder e o encoder criados acima

imagens_codificadas = encoder.predict(previsores_teste)
imagens_decodificadas = autoencoder.predict(previsores_teste)

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
    