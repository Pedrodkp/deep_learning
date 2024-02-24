from minisom import MiniSom
import pandas as pd

#aqui queremos fazer um mapa com as classes de vinho em um processo de aprendizagem nao supervisionada

base = pd.read_csv('wines.csv')
#ilocs ja usado N vezes, mas a primeira parte e o range de linhas, a segunda range de colunas
X = base.iloc[:,1:14].values #o ZERO e a classe, ele removemos
y = base.iloc[:,0].values #apenas as classes

from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range=(0,1))
X = normalizador.fit_transform(X)

##### MAPA AUTO ORG #####

#resolucao do mapa X e Y, encontrado com a formula:
#   √(5*√N) 
#sendo N a quantidade de linhas de dados, ou seja
# 5√178 registros = 5*13,11 = 65,65, raiz mais proxima arredondada 64, √64 = 8, logo 8X8
#input_len = quantidade de atributos
#sigma = equivale ao raio do PMU em neuronios, 1 e um neuronio
#learning_rate = atualizaçao dos registros do PMU mais elementos em volta com base nele
#random_seed = colocamos para sempre ter o mesmo resultado se rodar de novo, se ficar aleatorio o seed o resultado tbm fica para cada exec
som = MiniSom(x=8, y=8, input_len=13, sigma=1.0, learning_rate=0.5, random_seed=2) 

som.random_weights_init(X)

#num_interation equivale a epochs
som.train_random(data = X, num_iteration=100)

##### VER o MID #####
#mostra pontos adicionados
som._weights
#mostra proximidades
som._activation_map
#mostra matriz dos neuronios de fato
q = som.activation_response(X)

from pylab import pcolor, colorbar
#som.distance_map(), matriz de distancia, calcula o MID - mean inter neuron distance
#o .T inverte colunas com linhas (matriz transposta)
pcolor(som.distance_map().T)
#quanto mais escuro menor o numero, menor a distancia, mais claro maior a distancia do neuronio
#quanto mais longe menos confiavel
colorbar() 

#qual neuronio ganhador de cada registro, seleciona o BMU - best mach unit (neuronio VS registro)
#aqui vamos ver do registro UM, valores de X e Y no mapa
print(som.winner(X[1])) #3,6 verde claro
#outra linha
print(som.winner(X[2])) #3,2 verde medio

markers = ['o', 's', 'D'] #nomes que vamos dar a valores da variavel Y (1, 2, 3)
colors = ['r', 'g', 'b'] #cores que vamos dar a valores da variavel Y (1, 2, 3)
#valores estao em 1, 2, 3 entao alteramos para 0, 1, 2 para termos array
#ps: apenas um jeito diferente de fazer isso abaixo, poderia ser um lambda ou looping, mais para lembrar que existe esse jeito
y[y == 1] = 0
y[y == 2] = 1
y[y == 3] = 2

for i, x in enumerate(X):
    print(i) #id de registro
    print(x) #linha completa como s 13 atributos
    print(som.winner(x)) #BMU de cada registro
    
from pylab import plot
#montamos o grafico de novo para marcar
#marcacao
#bola na cor vermelha equivale a classe zero e por ai vai
for i, x in enumerate(X):
    w = som.winner(x)
    # o + 0.5 e apenas para centralizar no meio de quadrado o ponto
    # markeredgwidth e borda
    plot(w[0] +0.5, w[1]+0.5, markers[y[i]], markerfacecolor=None, markersize=10, markeredgecolor=colors[y[i]], markeredgewidth=2) 