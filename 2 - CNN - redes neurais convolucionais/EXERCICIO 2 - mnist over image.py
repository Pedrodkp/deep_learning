import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

#MNIST has that order: 0,1,2,3,4,5,6,7,8,9

(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()

previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 28, 28, 1)
previsores_treinamento[0]
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)

previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

previsores_treinamento /= 255
previsores_teste /= 255
previsores_treinamento[0]

classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape=(28, 28, 1), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))
classificador.add(Conv2D(32, (3,3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))
classificador.add(Flatten())
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=10, activation='softmax'))
classificador.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size=128, epochs=5, validation_data=(previsores_teste, classe_teste))
#resultado:
# 1st value: LOSS, which indicates how well your model is performing. 
#            It represents the error between the actual output and the predicted output for the test dataset. 
#            Lower values indicate better performance, as it means the model's predictions are closer to the actual values.
# 2nd value: ACCURACY, which indicates the proportion of correctly classified images in the test dataset. 
#            It represents the percentage of images for which the model's predictions match the actual labels. 
#            Higher values indicate better performance, with 1.0 being perfect accuracy.
resultado = classificador.evaluate(previsores_teste, classe_teste)

import numpy as np
from keras.preprocessing import image
import cv2

### PREVENDO DO ARQUIVO

def test_image_28(img):
    # Load the image
    img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # Resize the image to 28x28
    img_resized = cv2.resize(img_array, (28, 28))
    # Reshape the image to match the input shape expected by the model
    img_reshaped = img_resized.reshape(1, 28, 28, 1)
    # Normalize the pixel values
    img_normalized = img_reshaped / 255.0
    # Make prediction
    prediction = classificador.predict(img_normalized)
    print(img+" -> "+str(prediction))
    resultado = np.argmax(prediction)
    print(resultado)

test_image_28('digitos/test/5.jpg')
test_image_28('digitos/test/5.png')
test_image_28('digitos/test/7.jpeg')
test_image_28('digitos/test/8.jpg')

### PREVENDO CONTRA O MODELO
from_model = 100

plt.imshow(X_teste[from_model], cmap = 'gray')
plt.title('Classe ' + str(y_teste[from_model]))

# Criamos uma única variável que armazenará a imagem a ser classificada e
# também fazemos a transformação na dimensão para o tensorflow processar
imagem_teste = X_teste[from_model].reshape(1, 28, 28, 1)

# Convertermos para float para em seguida podermos aplicar a normalização
imagem_teste = imagem_teste.astype('float32')
imagem_teste /= 255

# Fazemos a previsão, passando como parâmetro a imagem
# Como temos um problema multiclasse e a função de ativação softmax, será
# gerada uma probabilidade para cada uma das classes. A variável previsão
# terá a dimensão 1, 10 (uma linha e dez colunas), sendo que em cada coluna
# estará o valor de probabilidade de cada classe
previsoes = classificador.predict(imagem_teste)

# Como cada índice do vetor representa um número entre 0 e 9, basta agora
# buscarmos qual é o maior índice e o retornarmos. Executando o código abaixo
# você terá o índice 7 que representa a classe 7
import numpy as np
resultado = np.argmax(previsoes)
print(resultado)
