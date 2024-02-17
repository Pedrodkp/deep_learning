from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

classificador = Sequential()
#input_shape forca uma resolucao, ou seja, o banco esta com N diferentes dimensoes, ele normaliza
classificador.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

alta_performance = True
if alta_performance:
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
classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#evita carregar formatos em disco
gerador_treinamento = ImageDataGenerator(rescale=1./255, rotation_range=7,horizontal_flip=True,shear_range=0.2,height_shift_range=0.07,zoom_range=0.2)
gerador_teste = ImageDataGenerator(rescale=1./255)

base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set', target_size = (64,64), batch_size=32, class_mode='binary')

base_teste = gerador_teste.flow_from_directory('dataset/test_set', target_size=(64,64), batch_size=32, class_mode = 'binary')

#steps_per_epoch = pode ter o tamanho do banco de imagens para maior precisao, no caso 4000
#validation_steps = pode ser 1000 para variar menos os pesos
classificador.fit_generator(base_treinamento, epochs=100, validation_data=base_teste)

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
    previsao_str = "gato" if previsao > 0.5 else "cachorro"
    print(img+" -> "+previsao_str+" - "+str(previsao))
    
test_image('dataset/test_set/gato/cat.3500.jpg')
test_image('dataset/test_set/cachorro/dog.3500.jpg')
test_image('test/kayla.jpeg')
test_image('test/kayla2.jpeg')
test_image('test/pacoca_pipoca.jpeg')
test_image('test/pacoca.jpeg')

classificado_json = classificador.to_json()
with open('classificador_cachorrogato.json', 'w') as json_file:
    json_file.write(classificado_json)
    
classificador.save_weights('classificador_cachorrogato.h5')
