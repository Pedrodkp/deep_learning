import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

#ja contem as atributos e classes
base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

#para evitar o erro (ValueError: Shapes (None, 1) and (None, 3) are incompatible) no fit
#transforma o atributo categorico em numerico
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
classe = label_encoder.fit_transform(classe)
#agora que temos 3 valores numericos, precisamos ter um "nome" numerico,  algo como
#iris setosa 0 0 1
#iris versicolor 0 1 0
#iris virginica 1 0 0
classe_dummy = np_utils.to_categorical(classe)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)

classificador = Sequential()
#units = qtde entr (4) + qtde saida (Iris-versicolor,Iris-setosa,Iris-virginica) / 2 = 3.5 = aprox 4
classificador.add(Dense(units=4, activation='relu', input_dim=4))
classificador.add(Dense(units=4, activation='relu'))
#3 neoronios na camada de saida por 3 classes na saida
#utiliza softmax para mais de duas classes na saida
classificador.add(Dense(units=3, activation='softmax'))

classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics='categorical_accuracy')

classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 1000)

resultado = classificador.evaluate(previsores_teste, classe_teste)
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

#para gerar a matrix precisamos de novo de uma unica coluna com valor, entao fazemos o 001 virar um indice
import numpy as np
classe_teste2 = [np.argmax(t) for t in classe_teste]
previsoes2 = [np.argmax(t) for t in previsoes]
from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(previsoes2, classe_teste2)