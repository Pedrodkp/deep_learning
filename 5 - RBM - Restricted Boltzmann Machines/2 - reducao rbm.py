import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM #implemetacao RBM no sklearn
from sklearn.naive_bayes import GaussianNB #estimativas de probabilidades
from sklearn.pipeline import Pipeline #executa varios processos em conjunto

#obj: reduzir a dimensionalidade do MNIST, basicamente reduzir a resolucacao com IA para destacar mais cada pixel importante

base = datasets.load_digits() #carrega o MNIST (base de numeros) mas reduzida
previsores = np.asarray(base.data, 'float32') #o ,64 sao os pixels, logo sao imagens 8x8
classe = base.target

normalizador = MinMaxScaler(feature_range=(0,1))
previsores = normalizador.fit_transform(previsores)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.2, random_state=0)

rbm = BernoulliRBM(random_state=0)
rbm.n_inter = 25 #num de epocas
rbm.n_components = 50 #num neuronios na camada escondida
naive_rbm = GaussianNB()
classificador_rbm = Pipeline(steps=[('rbm', rbm),('naive',naive_rbm)])
classificador_rbm.fit(previsores_treinamento, classe_treinamento)

plt.figure(figsize=(20,20)) #tamanho das imagens que vou mostrar
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i+1) #nao e a resolucao, apenas o tamanho (subplot coloca graficos dentro do grafico principal)
    plt.imshow(comp.reshape((8,8)), cmap=plt.cm.gray_r) #imagem em si, cinza
    plt.xticks(()) #tira os caption do X
    plt.yticks(()) #tira os caption do Y
plt.show() #mostra imagens (numeros) com reducao da dimensionalidade

##### RBM vs no RBM #####
#obj: fazer isso ajuda? reduzir a dimensionalidade

previsoes_rbm = classificador_rbm.predict(previsores_teste) #cada registro e submetido para classificador para determinar classe por probabilidade (classes de 1 a 9 no caso)
precisao_rbm = metrics.accuracy_score(previsoes_rbm, classe_teste) #83%

#execucao sem RBM
#e exatamente o mesmo codigo, mas rodando o naive sem rbm
naive_simples = GaussianNB()
naive_simples.fit(previsores_treinamento, classe_treinamento)
previsoes_naive = naive_simples.predict(previsores_teste)
precisao_naive = metrics.accuracy_score(previsoes_naive, classe_teste) #81% (piorou, para IA e muita coisa 2%, vale o ganho)