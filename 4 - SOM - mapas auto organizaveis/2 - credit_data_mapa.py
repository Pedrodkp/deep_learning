from minisom import MiniSom
import pandas as pd
import numpy as np

#objetivo, tentar identificar clientes com chance de cometer fraude

#default column, tendencia de fraude
base = pd.read_csv('credit_data.csv')
base = base.dropna()

#existem pessoas com idade negativa, vamos substituir pela media
age_mean = base['age'].mean()
base.loc[base.age < 0, 'age'] = age_mean

##### IMPORTANTE: o client_id nao deveria entrar aqui, feito dessa forma apenas para carater de explicacao
x = base.iloc[:, 0:4].values
y = base.iloc[:, 4].values

from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range=(0,1))
X = normalizador.fit_transform(x)

#1997 reg = √(5*√1997) = √(5*44,68) = √223 = 15 (aprox)
som = MiniSom(x=15, y=15, input_len=4, random_seed=0)
som.random_weights_init(X)
som.train_random(data=X,num_iteration=100)

#nesse caso os registros mais distantes, perto de 1 ou amarelo seriam os mais outliers, ou com mais chance de cometer fraude
from pylab import pcolor, colorbar, plot
pcolor(som.distance_map().T)
colorbar()

#basicamente na representacao de neuronois X linhas, os registros que escolheram os neuronios mais distantes tendem a ser os mais diferentes
#como nesse caso os clientes que registram fraude sao muito diferentes dos que nao fazem isso, eles escolher esses neuronios
#   neuronios bola sao os que tiveram o credito aprovado
#   neuronios quadrado sao os que tiveram credito negado
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] +0.5, w[1]+0.5, markers[y[i]], markerfacecolor=None, markersize=5, markeredgecolor=colors[y[i]], markeredgewidth=2) 
    
##### SELECAO DE REGISTROS #####
#pegamos os registros que o mapa indica poder ser fraude para analise humana
    
mapeamento = som.win_map(X) #pega registros agrupados por posicao (neuronios)
suspeitos = np.concatenate((mapeamento[(4,5)], mapeamento[(6,13)]), axis=0) #visualmente pegamos os neuronois que queremos, no caso os mais amarelos
suspeitos = normalizador.inverse_transform(suspeitos) #transformacao invertida

##### IMPORTANTE: o client_id nao deveria entrar aqui, poderiamos entao usar um HASH MD5 de todos os atributos 
#                 e comparar para coletar os suspeitos na base 
classe = []
for i in range(len(base)):
    for j in range(len(suspeitos)):
        if base.iloc[i, 0] == int(round(suspeitos[j,0])):
            classe.append(base.iloc[i,4])
            
classe = np.asarray(classe) #ZERO, suspeitos que ganharam emprestimo e UM suspeitos que nao ganharam
suspeitos_final = np.column_stack((suspeitos, classe)) #concatena
suspeitos_final = suspeitos_final[suspeitos_final[:,4].argsort()] #ordenado pela coluna 4
