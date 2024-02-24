from minisom import MiniSom
import pandas as pd
import numpy as np

#objetivo, tentar identificar registros que nao sao interpretados nem como homer nem como bart

base = pd.read_csv('../2 - CNN - redes neurais convolucionais/homer bart/personagens.csv')
base = base.dropna()

x = base.iloc[:, 0:6].values
y = base.iloc[:, 6].values

from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range=(0,1))
X = normalizador.fit_transform(x)

#293 reg = √(5*293) = √(5*17,1) = √85,58 = 9,25 = 9 (aprox)
som = MiniSom(x=9, y=9, input_len=6, random_seed=0)
som.random_weights_init(X)
som.train_random(data=X,num_iteration=1000)

#nesse caso os registros mais distantes, perto de 1 ou amarelo seriam os menos parecidos com homer ou bart
from pylab import pcolor, colorbar, plot
pcolor(som.distance_map().T)
colorbar()

#basicamente na representacao de neuronois X linhas, os registros que escolheram os neuronios mais distantes tendem a ser os mais diferentes
#o for Bart
#s for Homer
markers = ['o', 's']
colors = ['r', 'g']
y_numerico = np.where(y == 'Homer', 1, np.where(y == 'Bart', 0, -1))

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y_numerico[i]], markerfacecolor=None, markersize=5, markeredgecolor=colors[y_numerico[i]], markeredgewidth=2)
    
##### SELECAO DE REGISTROS #####
#pegamos os registros que estao muito diferentes
    
mapeamento = som.win_map(X) #pega registros agrupados por posicao (neuronios)
suspeitos = (mapeamento[(7,4)]) #visualmente pegamos os neuronois que queremos, no caso os mais amarelos
suspeitos = normalizador.inverse_transform(suspeitos) #transformacao invertida

print("Registros suspeitos:")
print(suspeitos)

##### ENCONTRANDO NA BASE ORIGINAL #####
base_rounded = base.iloc[:, :6].round(decimals=4) #apenas os registros que podemos comparar, arredondamos por causa do float
suspeitos_rounded = np.round(suspeitos[:, :6], decimals=4)  # Round only the first 6 columns#apenas os registros que podemos comparar, arredondamos por causa do float

# procuramos os registros que existem em ambos
matching_records = []
matching_indices = []
for i, s in enumerate(suspeitos_rounded):
    for j, row in base_rounded.iterrows():
        if np.array_equal(row.values, s):
            matching_records.append(base.iloc[j])
            matching_indices.append(j)

print("Registros suspeitos na base arredondada:")
for index, record in zip(matching_indices, matching_records):
    print(f"Index: {index}")
    print(record)