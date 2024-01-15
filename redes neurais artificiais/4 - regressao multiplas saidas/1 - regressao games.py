import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model

#a ideia aqui e prever 3 valores de uma vez, o preco nos EUA, JP e EURO

base = pd.read_csv('games.csv')

#remover o que nao vamos utilizar
base = base.drop('Other_Sales', axis=1)
#se quisesse prever as vendas totais, poderiamos fazer com o Global_Sales igual o modulo anterior
base = base.drop('Global_Sales', axis=1)
#como tem o publisher vale a pena trabalhar com ele
base = base.drop('Developer', axis=1)

#deveriamos preencher valores faltantes, mas nao podemos usar a media aqui e nao vamos buscar
#entao vamos apagar para fazer o modulo
base = base.dropna(axis=0)
#ficar com o que teve mais de uma venda
base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]

#o nome nao faz sentido, por que a ideia e prever as vendas com base nas caracteristicas do jogo
#o nome ja determina por historico, ai poderiam ser feito apenas com jogos da mesma franquia para um novo
#ex: trabalhar so com jogos que tem "Mario" no nome
base['Name'].value_counts()
nome_jogos = base.Name
base = base.drop('Name', axis=1)

#separamos os atributos dos valores
previsores = base.iloc[:, [0,1,2,3,7,8,9,10,11]].values
#separamos os valores de saida
venda_na = base.iloc[:, 4].values
venda_eu = base.iloc[:, 5].values
venda_jp = base.iloc[:, 6].values

#tranformar textos em valores
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:,0])
previsores[:, 2] = labelencoder.fit_transform(previsores[:,2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:,3])
previsores[:, 8] = labelencoder.fit_transform(previsores[:,8])

#agora vamos criar colunas para cada valor, dummy 0=000
from sklearn.compose import ColumnTransformer
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,2,3,8])],remainder='passthrough') 
previsores = onehotencoder.fit_transform(previsores).toarray()

#vamos utilizar outro model que permite redes mais robustas
#shape=formato (61 atributos)
camada_entrada = Input(shape=(61,))
#61 entradas + 3 saidas /2 = 32, testar sigmoid OU relu
camada_oculta1 = Dense(units=32, activation='sigmoid')(camada_entrada)
camada_oculta2 = Dense(units=32, activation='sigmoid')(camada_oculta1)
#vamos ter uma camada para cada preco em cada pais em vez de 3 neuronios
#na
camada_saida1 = Dense(units=1, activation='linear')(camada_oculta2)
#eu
camada_saida2 = Dense(units=1, activation='linear')(camada_oculta2)
#jp
camada_saida3 = Dense(units=1, activation='linear')(camada_oculta2)

#rodar rede
regressor = Model(inputs=camada_entrada, outputs=[camada_saida1, camada_saida2, camada_saida3])
regressor.compile(optimizer = 'adam', loss='mse')
regressor.fit(previsores, [venda_na, venda_eu, venda_jp], epochs=5000, batch_size=100)

#nao fizemos validacao cruzada ou base treinamento/teste, mas deve-se fazer, apenas prova de conceito
#agora podemos comparar (previsao_na VS vendas_na) e por ai vai
previsao_na, previsao_eu, previsao_jp = regressor.predict(previsores)
