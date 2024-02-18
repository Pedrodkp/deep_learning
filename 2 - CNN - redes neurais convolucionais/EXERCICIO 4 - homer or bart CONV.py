from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

classificador = Sequential()

classificador.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Flatten())
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units=1, activation='sigmoid'))

classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

gerador_treinamento = ImageDataGenerator(rescale=1./255, rotation_range=7, horizontal_flip=True, shear_range=0.2, height_shift_range=0.07, zoom_range=0.2)
gerador_teste = ImageDataGenerator(rescale=1./255)

base_treinamento = gerador_treinamento.flow_from_directory('homer bart/dataset_personagens/training_set', target_size=(64,64), batch_size=32, class_mode='binary')
base_teste = gerador_treinamento.flow_from_directory('homer bart/dataset_personagens/test_set', target_size=(64,64), batch_size=32, class_mode='binary')

steps_per_epoch = len(base_treinamento)
validation_steps = len(base_teste)
classificador.fit_generator(base_treinamento, steps_per_epoch=steps_per_epoch, epochs=100, validation_data=base_teste, validation_steps=validation_steps)
previsoes = classificador.predict(base_teste)

import numpy as np
import keras.utils as image

def test_image(img):
    imagem_teste = image.load_img(img, target_size=(64,64))
    imagem_teste = image.img_to_array(imagem_teste)
    imagem_teste /= 255
    #converte formato do tensorflow (1,64,64,3) sendo 1=quantidade de imagens, 64x64 resolucao, 3 por ter cor (rgb)
    imagem_teste = np.expand_dims(imagem_teste, axis=0)
    
    previsao = classificador.predict(imagem_teste)
    print(img+" -> "+str(previsao))
    
test_image('homer bart/dataset_personagens/homer.png')
test_image('homer bart/dataset_personagens/bart.png')
test_image('homer bart/dataset_personagens/bart2.png')
test_image('homer bart/dataset_personagens/bart3.png')
test_image('homer bart/dataset_personagens/bart4.png')

#this works, near 1 will be homer, near ZERO be bart, be carefult about values like 4.3939377e-05 because it is equivalente to 0.000043939377, so bart
