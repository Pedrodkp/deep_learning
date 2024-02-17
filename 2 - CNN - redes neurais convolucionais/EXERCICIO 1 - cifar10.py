import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

#a ideia aqui e classificar usando a cifar-10
#https://www.cs.toronto.edu/~kriz/cifar.html
#a base contemplata (exatamente nessa ordem): 
#avião, automóvel, pássaro, gato, veado, cachorro, sapo, cavalo, barco e caminhão

(X_treinamento, y_treinamento), (X_teste, y_teste) = cifar10.load_data()

# Mostra a imagem e a respectiva classe, de acordo com o índice passado como parâmetro
# Você pode testar os seguintes índices para visualizar uma imagem de cada classe
# Avião - 650
# Automóvel - 4
# Pássaro - 6
# Gato - 9
# Veado - 3
# Cachorro - 813
# Sapo - 651
# Cavalo - 652
# Barco - 811
# Caminhão - 970
plt.imshow(X_treinamento[2])
plt.title('Classe '+ str(y_treinamento[2]))

previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 32, 32, 3)
previsores_teste = X_teste.reshape(X_teste.shape[0], 32, 32, 3)
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')
previsores_treinamento /= 255
previsores_teste /= 255
classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape=(32, 32, 3), activation='relu'))
classificador.add(MaxPooling2D(pool_size=(2,2)))
classificador.add(Flatten())
classificador.add(Dense(units=256, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=256, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=10, activation='softmax'))
classificador.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size=256, epochs=10, validation_data=(previsores_teste, classe_teste))
resultado = classificador.evaluate(previsores_teste, classe_teste)

import numpy as np
import keras.utils as image

def test_image(img):
    imagem_teste = image.load_img(img, target_size=(32,32))
    imagem_teste = image.img_to_array(imagem_teste)
    imagem_teste /= 255
    imagem_teste = np.expand_dims(imagem_teste, axis=0)
    previsao = classificador.predict(imagem_teste)
    print(img+" -> "+str(previsao))
    
test_image('gatos cachorros/test/kayla.jpeg')
test_image('gatos cachorros/test/kayla2.jpeg')
test_image('gatos cachorros/test/pacoca_pipoca.jpeg')
test_image('gatos cachorros/test/pacoca.jpeg')
test_image('gatos cachorros/test/ipanema.jpg')
test_image('gatos cachorros/test/opala.jpg')
test_image('gatos cachorros/test/amazonas.jpg')
