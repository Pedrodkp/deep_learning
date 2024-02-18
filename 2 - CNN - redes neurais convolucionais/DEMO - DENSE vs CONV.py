## IMPORTAÇÕES

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

## CRIAÇÃO DA REDE NEURAL:
    
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
classificador.add(Dense(units=1 , activation='sigmoid'))

classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

## GERADORES

gerador_treinamento = ImageDataGenerator(rescale=1./255,
                                         rotation_range=7,
                                         horizontal_flip=True,
                                         shear_range=0.2,
                                         height_shift_range=0.07,
                                         zoom_range=0.2)

gerador_teste = ImageDataGenerator(rescale=1./255)

base_treinamento = gerador_treinamento.flow_from_directory('homer bart/dataset_personagens/training_set',
                                                                      target_size=(64,64),
                                                                      batch_size=10,
                                                                      class_mode='binary')  

base_teste = gerador_teste.flow_from_directory('homer bart/dataset_personagens/test_set',
                                            target_size = (64, 64),
                                            batch_size = 10,
                                            class_mode = 'binary')

base_treinamento.class_indices #bart = 0 / homer= 1

## TREINAMENTO

classificador.fit_generator(base_treinamento, steps_per_epoch=196,
                            epochs=200, validation_data=base_teste,
                            validation_steps=73)

## PREVISÃO COM UMA IMAGEM

import numpy as np
import keras.utils as image

img_test = image.load_img('homer bart/dataset_personagens/homer2.jpg', target_size=(64,64))
img_test = image.img_to_array(img_test)
img_test /= 255
img_test = np.expand_dims(img_test, axis=0)
previsao_img_test= classificador.predict(img_test)

if previsao_img_test >= 0.5:
    print(f'Imagem classificada como Homer.'
          f'\nValor da precisão: {previsao_img_test}'
          f'\nOBS: {base_treinamento.class_indices}')
else:
    print('Imagem clssificada como Bart.'
          f'\nValor da precisão: {previsao_img_test}'
          f'\nOBS: {base_treinamento.class_indices}')


img_test2= image.load_img('homer bart/dataset_personagens/bart5.jpg', target_size=(64,64))
img_test2 = image.img_to_array(img_test2)
img_test2 /=255
img_test2 = np.expand_dims(img_test2, axis=0)
previsao_img_test2 = classificador.predict(img_test2)

if previsao_img_test2 >= 0.5:
    print(f'Imagem classificada como Homer.'
          f'\nValor da precisão: {previsao_img_test2}'
          f'\nOBS: {base_treinamento.class_indices}')
else:
    print('Imagem clssificada como Bart.'
          f'\nValor da precisão: {previsao_img_test2}'
          f'\nOBS: {base_treinamento.class_indices}')