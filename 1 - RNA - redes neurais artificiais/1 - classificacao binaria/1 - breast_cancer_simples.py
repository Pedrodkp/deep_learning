import pandas as pd

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

import keras
from keras.models import Sequential
from keras.layers import Dense #cada neoronio liga com todos os outros da camada subsequente

#a ideia aqui e fazer um treinamento simples para pegar cada conceito

classificador = Sequential()
#nao tera camada de entrada
#units = numero de neoronios, (30 entradas + 1 saida (benigno / maligno)) / 2 = 15.5, aprox 16
#activation = funcao de ativacao, comeca com a 'relu' para experimentar
#kernel_initializer = inicializacao dos pesos
#input_dim = elementos na camada de entrada, assim nao precisa da primeira camada, ele ja cria uma com 30 neuronios de entrada
classificador.add(Dense(units= 16, activation= 'relu', \
                        kernel_initializer= 'random_uniform', input_dim = 30)) #primeira camada oculta 
    
#posso SE QUISER (funciona com uma) criar mais camadas apenas declarando ela, para maior precisao
#pode ser a mesma quantidade, menos ou mais neoronios
classificador.add(Dense(units= 16, activation= 'relu', \
                        kernel_initializer= 'random_uniform')) #primeira camada oculta 
    
#camada de saida
#units = 1 neuronio = benigno (prox ZERO) / maligno (prox UM)
#activation = funcao de ativacao, usa a sigmoid para probabilidade ZERO-UM
classificador.add(Dense(units= 1, activation='sigmoid'))

otimizador = 'adam'
if False: #posso calibrar o otimizador
    #calibrando o otimizador
    #lr (learning rate): para encontrar o maximo global, define o tamanho dos passos de aprendizagem
    #decay: incremento do lr, funciona como uma reducao, ex passo 1, depois passo 0.5, depois 0.25
    #   para ganhar velocidade, ja que o lr pode ser grande no comeco*, e curto no meio para o fim
    # * onde a resposta e mais inprovavel no comeco, e mais provavel do meio pro fim
    #clipvalue: caso chegue em algo perto de aceito, congela uma margem para evitar ping/pong de testes entre valores
    ####OLD
    #otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
    ###NEW
    import tensorflow as tf
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
#compila a rede
#optimizer = adam, para decida do gradiente
#loss = binary_crossentropy, funcao de perda mais comum, leva em consideracao um logaritimo
#binary_crossentropy* penaliza mais classificacao errada
#metrics:
    # - binary_accuracy, certos X errados
classificador.compile(optimizer=otimizador, loss='binary_crossentropy', \
                      metrics= ['binary_accuracy'])    
    
#relaciona previsores com classes e define a cada quantos registros atualiza os pesos, tambem define quantas epocas
#nao pode levar em consideracao pois esta usando o treinamento como teste aqui
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs=100)

#aqui e possivel ver os pesos para cada camada, treinamento de fato
#o primero valor de cada linha 30, e o numero de entradas, o segundo 16 e a qtd de neoronios
#a segunda linha e a quantidade de bias, no caso ele criou mais um e ligou com os 16 da camada
#   acontece por que em Dense(), existe um argumento use_bias=True (default)
pesos0 = classificador.layers[0].get_weights()
print(pesos0)

#tem 16 em vez de 30, por serem 16 neoronios de saida da primeira camada
pesos1 = classificador.layers[1].get_weights()
print(pesos1)

#tem 1, por que e apenas um neoronio na camada de saida
pesos2 = classificador.layers[2].get_weights()
print(pesos2)

#aqui faz o teste cego
previsoes = classificador.predict(previsores_teste)

#visualmente pode comparar o classe_teste com previsoes
#converte em boolean
previsoes = (previsoes > 0.5)

#pode se ter uma media de acerto
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)

#bem como uma matriz, no caso quantos zeros foram classificados como zero mesmo, ou como 1 e vice versa
#Y - sao os valores que sabemps
#X - e a resposta calculada
matriz = confusion_matrix(classe_teste, previsoes)

#mesma coisa com keras, da o erro e o percentual de acerto
resultado = classificador.evaluate(previsores_teste, classe_teste)