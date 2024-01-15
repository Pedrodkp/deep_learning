import pandas as pd

#a ideia aqui agora nao e mais classificar algo, e sim prever algo
#a resposta nao necessariamente precisar existir prevista

#analiser os atributos e remover os que nao impactam no preco
base = pd.read_csv('autos.csv', encoding='ISO-8859-1')
base = base.drop('dateCrawled', axis=1)
base = base.drop('dateCreated', axis=1)
base = base.drop('nrOfPictures', axis=1)
base = base.drop('postalCode', axis=1)
base = base.drop('lastSeen', axis=1)

#o ideial e ver todos os atributos com o values_counts um por um
#e analiser se nao vale a pena remover os que sao muito desproporcionais
#ou remover o atributo, ou os registros com os valores muito descripantes

#vemos que o name nao e padronizado, vamos usar o modelo
base['name'].value_counts()
base = base.drop('name', axis=1)

#como so tem 3 vendedores de empresa, vamos ignorar pois nao tem percentual relevanet
base['seller'].value_counts()
base = base.drop('seller', axis=1)

#os de leilao "Gesuch", tambem sao poucos, vamos desconsiderar para nao penalizar o preco
base['offerType'].value_counts()
base = base.drop('offerType', axis=1)

#tambem precisamos remover valores inconsistentes, ou seja, o atributo deve existir mas esta ruim o valor
i1 = base.loc[base.price <= 10]
media_de_precos = base.price.mean() #ou usar e media, mas nao faz sentido nesse caso
base = base[base.price > 10]

#tambem checar valores muito grandes
i2 = base.loc[base.price > 350000]
base = base[base.price <= 350000]

#tratar tbm valores nulos
base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts() #pegar o que mais aparece: limousine
base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts() #manuell
base.loc[pd.isnull(base['model'])]
base['model'].value_counts() #golf
base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts() #benzin
base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts() #nein

#null por
valores = {'vehicleType': 'limousine',
           'gearbox': 'manuell',
           'model': 'golf',
           'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}
base = base.fillna(value = valores)

previsores = base.iloc[:, 1:13].values
#nao chama de classe, pois nao estamos trabalhando com problemas de classificaçao
preco_real = base.iloc[:, 0].values

#transforma de categoricos para numericos 'texto para numeros'
print(previsores[0])
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsores = LabelEncoder()
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 10] = labelencoder_previsores.fit_transform(previsores[:, 10])
print(previsores[0])


#agora previsamos mudar os valores decimais para dummy, ou seja, se tem 10 categorias na coluna
#agora precisa virar 10 atributos (colunas) tipo true/false
#OneHotEncoder pode ser usado para dados em que uma categoria nao e maior que a outra
#ex: vc nao pode falar que o cambio manual e "maior" que o automatico, por ser uma preferencia
from sklearn.compose import ColumnTransformer
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,3,5,8,9,10])],remainder='passthrough') 
previsores = onehotencoder.fit_transform(previsores).toarray()

#comeca a construir a rede neuronal
from keras.models import Sequential
from keras.layers import Dense

regressor = Sequential()
#units 316 entradas + 1 saida (valor do carro) / 2 = 158
#input_dim=316 por causa do hoencoder
regressor.add(Dense(units=1s58, activation='relu', input_dim=316))
regressor.add(Dense(units=158, activation='relu'))
#activation=linear, o valor der vamos usar de preco, nao queremos convergir para um binario
regressor.add(Dense(units=1, activation='linear'))
regressor.compile(loss='mean_absolute_error', optimizer='adam', metrics='mean_absolute_error')
regressor.fit(previsores,preco_real, batch_size=300, epochs=100)

#agora fazemos a previsao de precos
previsoes = regressor.predict(previsores)

#agora e possivel comparar (previsoes VS preco_real)
print(f"{previsoes[0]}{preco_real[0]}")
#tambem podemos comparar media prevista com real
preco_real.mean()
previsoes.mean()
