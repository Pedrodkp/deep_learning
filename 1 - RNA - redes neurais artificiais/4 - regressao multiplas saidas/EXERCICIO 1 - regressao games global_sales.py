import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor

base = pd.read_csv('games.csv')

base = base.drop('Other_Sales', axis=1)
base = base.drop('Developer', axis=1)
base = base.drop('NA_Sales', axis=1)
base = base.drop('EU_Sales', axis=1)
base = base.drop('JP_Sales', axis=1)

base = base.dropna(axis=0)
base = base.loc[base['Global_Sales'] > 1]

base['Name'].value_counts()
nome_jogos = base.Name
base = base.drop('Name', axis=1)

previsores = base.iloc[:, [0,1,2,3,5,6,7,8,9]].values
venda_global = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:,0])
previsores[:, 2] = labelencoder.fit_transform(previsores[:,2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:,3])
previsores[:, 8] = labelencoder.fit_transform(previsores[:,8])

from sklearn.compose import ColumnTransformer
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,2,3,8])],remainder='passthrough') 
previsores = onehotencoder.fit_transform(previsores).toarray()

def criar_rede():
    #shape=formato 99 atributos)
    camada_entrada = Input(shape=(99,))
    #99 entradas + 1 saidas /2 = 50, testar sigmoid OU relu
    camada_oculta1 = Dense(units=50, activation='sigmoid')(camada_entrada)
    drop_out1 = Dropout(0.5)(camada_oculta1)
    camada_oculta2 = Dense(units=50, activation='sigmoid')(drop_out1)
    drop_out2 = Dropout(0.5)(camada_oculta2)
    camada_saida = Dense(units=1, activation='linear')(drop_out2)
    
    #rodar rede
    regressor = Model(inputs=camada_entrada, outputs=[camada_saida])
    regressor.compile(optimizer = 'adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
    return regressor

def custom_scoring_function(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

regressor = KerasRegressor(build_fn=criar_rede, epochs=5000, batch_size=100)
resultados = cross_val_score(estimator=regressor, X=previsores, y=venda_global, cv=10,scoring=custom_scoring_function)

media_previsao_global = resultados.mean()
desvio = resultados.std()
media_venda_global = venda_global.mean()