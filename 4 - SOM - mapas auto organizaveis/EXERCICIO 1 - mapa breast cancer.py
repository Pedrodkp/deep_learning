from minisom import MiniSom
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar, plot

#aqui queremos fazer um mapa com a classificacao de cancer

X = pd.read_csv('../1 - RNA - redes neurais artificiais/1 - classificacao binaria/entradas_breast.csv')
y = pd.read_csv('../1 - RNA - redes neurais artificiais/1 - classificacao binaria/saidas_breast.csv')

normalizador = MinMaxScaler(feature_range=(0,1))
X = normalizador.fit_transform(X)
y = y.iloc[:, 0]

##### MAPA AUTO ORG #####

#resolucao do mapa X e Y, encontrado com a formula:
#   √(5*√N) 
#sendo N a quantidade de linhas de dados, ou seja
# 5√568 registros = 5*23,83 = 119,16 raiz 119,16 e 10,91, logo 11x11
#input_len = quantidade de atributos
#sigma = equivale ao raio do PMU em neuronios, 1 e um neuronio
#learning_rate = atualizaçao dos registros do PMU mais elementos em volta com base nele
#random_seed = colocamos para sempre ter o mesmo resultado se rodar de novo, se ficar aleatorio o seed o resultado tbm fica para cada exec
som = MiniSom(x=11, y=11, input_len=30, sigma=10.0, learning_rate=5, random_seed=2) 

som.random_weights_init(X)

#num_interation equivale a epochs
som.train_random(data = X, num_iteration=1000)

##### VER o MID #####
#mostra pontos adicionados
som._weights
#mostra proximidades
som._activation_map
#mostra matriz dos neuronios de fato
q = som.activation_response(X)

pcolor(som.distance_map().T)
colorbar() 

print(som.winner(X[1]))
print(som.winner(X[2]))

markers = ['o', 's']
colors = ['b', 'r']
   
for i, x in enumerate(X):
    w = som.winner(x)
    # o + 0.5 e apenas para centralizar no meio de quadrado o ponto
    # markeredgwidth e borda
    plot(w[0] +0.5, w[1]+0.5, markers[y[i]], markerfacecolor=None, markersize=5, markeredgecolor=colors[y[i]], markeredgewidth=2) 