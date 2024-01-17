from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

#a ideia aqui e aumentar o banco de dados de imagens, usando as imagens que ja existem

(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 28, 28, 1)
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')
previsores_treinamento /= 255
previsores_teste /= 255
classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape=(28, 28, 1), activation='relu'))
classificador.add(MaxPooling2D(pool_size=(2,2)))
classificador.add(Flatten())
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dense(units=10, activation='softmax'))
classificador.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#faz modificacoes na imagem que ja existe e modifica para aumentar a base
## https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
gerador_treinamento = ImageDataGenerator(rotation_range = 7, horizontal_flip= True, shear_range=0.2, height_shift_range=0.07, zoom_range=0.2)
#sem parametros nao ira fazer nada, mudar as de teste invalidaria a validacao
gerador_teste = ImageDataGenerator()

#aqui temos as novas bases
base_treinamento = gerador_treinamento.flow(previsores_treinamento, classe_treinamento, batch_size=128)
base_teste = gerador_teste.flow(previsores_teste, classe_teste, batch_size=128)

#as divisoes sao alguns numeros magicos para performance
classificador.fit_generator(base_treinamento, steps_per_epoch=60000 / 128, epochs=5, validation_data=base_teste, validation_steps=10000 / 128)
#ate melhora a precisao, mas seria necessario muito mais epochs, mas pode ser util para outras bases de dados