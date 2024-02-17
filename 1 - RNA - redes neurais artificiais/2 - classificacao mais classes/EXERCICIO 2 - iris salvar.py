import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
#tive que mudar para lib nova
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

#objetivo, aplicar o tunning, normal esta 0.9533333333333334, melhorar

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
classe = label_encoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

classificador = Sequential()
classificador.add(Dense(units=8, activation='relu', input_dim=4))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=8, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=3, activation='softmax'))
classificador.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
classificador.fit(previsores, classe, batch_size=10, epochs=500)

classificado_json = classificador.to_json()
with open('classificador_iris.json', 'w') as json_file:
    json_file.write(classificado_json)
    
classificador.save_weights('classificador_iris.h5')