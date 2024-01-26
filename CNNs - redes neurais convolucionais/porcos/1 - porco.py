from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

classificador = Sequential()
#input_shape forca uma resolucao, ou seja, o banco esta com N diferentes dimensoens, ele normaliza
classificador.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Flatten())
classificador.add(Dense(units=128,activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=128,activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=1,activation='sigmoid'))
#loss=binary_crossentropy por ser apenas gato/cachorros
classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

#evita carregar formatos em disco
gerador_treinamento = ImageDataGenerator(rescale=1./255)
gerador_teste = ImageDataGenerator(rescale=1./255)

base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set', target_size = (64,64), batch_size=32, class_mode='binary')
base_teste = gerador_teste.flow_from_directory('dataset/test_set', target_size=(64,64), batch_size=32, class_mode = 'binary')

#steps_per_epoch = pode ter o tamanho do banco de imagens para maior precisao, no caso 4000
#validation_steps = pode ser 1000 para variar menos os pesos
classificador.fit_generator(base_treinamento, epochs=10, validation_data=base_teste)

import numpy as np
import keras.utils as image

#opcoes
print("class indices:", base_treinamento.class_indices)

def test_image(img):
    imagem_teste = image.load_img(img, target_size=(64,64))
    imagem_teste = image.img_to_array(imagem_teste)
    imagem_teste /= 255
    #converte formato do tensorflow (1,64,64,3) sendo 1=quantidade de imagens, 64x64 resolucao, 3 por ter cor (rgb)
    imagem_teste = np.expand_dims(imagem_teste, axis=0)
    
    previsao = classificador.predict(imagem_teste)
    print(img+" -> "+str(previsao))
    
test_image('dataset/kayla.jpeg')
test_image('dataset/kayla2.jpeg')
test_image('dataset/pacoca_pipoca.jpeg')
test_image('dataset/pacoca.jpeg')
test_image('dataset/Captura de tela 2024-01-10 224720.png')
test_image('dataset/Captura de tela 2024-01-10 224810.png')
test_image('dataset/Captura de tela 2024-01-10 224734.png')
test_image('dataset/cat.14.jpg')
test_image('dataset/dog.10.jpg')

previsoes = classificador.predict(base_teste)

classificado_json = classificador.to_json()
with open('classificador_porco.json', 'w') as json_file:
    json_file.write(classificado_json)
    
classificador.save_weights('classificador_porco.h5')
