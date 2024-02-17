import tensorflow as tf
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

##a ideia aqui e usar toda a base de amostra como teste rotacionando
##em vez de fazer 75% treinamento e 25% testes, vamos dividir em 10 partes
##em 10 execucoues, 1 parte sera o teste, depois a segunda, depois a terceira...
##assim toda a base sera treinamento e teste, para nao perder nenhuma oporunitade de treinamento

def criar_rede():
    #zera alguns inputs para diminuir o everfitting, se exagerar tera underfitting, pode fazer por camada
    dropout = 0.2
    classificador = Sequential()
    
    classificador.add(Dense(units= 16, activation= 'relu', \
                            kernel_initializer= 'random_uniform', input_dim = 30)) #primeira camada oculta
    if dropout: 
        classificador.add(Dropout(dropout))
        
    classificador.add(Dense(units= 16, activation= 'relu', \
                            kernel_initializer= 'random_uniform')) #primeira camada oculta 
    if dropout: 
        classificador.add(Dropout(dropout))
        
    classificador.add(Dense(units= 1, activation='sigmoid'))
        
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    classificador.compile(optimizer=optimizer, loss='binary_crossentropy', \
                          metrics= ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn=criar_rede, epochs=100, batch_size=10)

#cv = quantas vezes vai fazer o teste
#aqui ele vai fazer 10 treinamentos
resultados = cross_val_score(estimator=classificador, X=previsores, y=classe, cv=10, scoring='accuracy')

#resultado final, elimina falsos positivos (alto acerto) e negativos (baixo acerto)
media = resultados.mean()

#desvio padrao, quanto maior o valor, maior o overfitting, ou seja, ela se comporta muito bem com os dados de treino
#mas tende a ter problemas com novos dados, ou seja, se adpta demais aos dados usados para treinamento
#ex: para ensinar o que e peixe, usa muitos peixes do mesmo tipo, nao aprende de fato nesse caso
#    e como a IA memorizando em vez de aprendendo, como um aluno que decora em vez de aprender
desvio = resultados.std()

#underfitting, substima o problema, matar t-rex com raquete, causado por resultados ruins na base de treinamento
#overfitting, superestima o problem, matar o mosquito com bazuca, resultados bons demais no treinamento, mas ruins no teste