import numpy as np
from keras.models import model_from_json

arquivo = open('classificador_iris.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador_iris.h5')

novo = np.array([[3.0, 8.34, 1.18, 9.00]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.75)
