import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor

base = pd.read_csv('autos.csv', encoding='ISO-8859-1')
base = base.drop('dateCrawled', axis=1)
base = base.drop('dateCreated', axis=1)
base = base.drop('nrOfPictures', axis=1)
base = base.drop('postalCode', axis=1)
base = base.drop('lastSeen', axis=1)
base = base.drop('name', axis=1)
base = base.drop('seller', axis=1)
base = base.drop('offerType', axis=1)
base = base[base.price > 10]
base = base[base.price <= 350000]

valores = {'vehicleType': 'limousine','gearbox': 'manuell','model': 'golf','fuelType': 'benzin','notRepairedDamage': 'nein'}
base = base.fillna(value = valores)

previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

labelencoder_previsores = LabelEncoder()
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 10] = labelencoder_previsores.fit_transform(previsores[:, 10])

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,3,5,8,9,10])],remainder='passthrough') 
previsores = onehotencoder.fit_transform(previsores).toarray()

def custom_scoring_function(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def criar_rede(loss):
    regressor = Sequential()
    regressor.add(Dense(units=158, activation='relu', kernel_initializer = 'random_uniform', input_dim=316))
    regressor.add(Dense(units=158, activation='relu', kernel_initializer = 'random_uniform'))    
    regressor.add(Dense(units=1, activation='linear'))
    regressor.compile(loss=loss, optimizer='adam', metrics=['mean_absolute_error'])
    return regressor

regressor = KerasRegressor(build_fn=criar_rede)
parametros = {'batch_size' : [100],
              'epochs' : [300],
              'loss' : ["mean_squared_error" , "mean_absolute_error" , "mean_absolute_percentage_error" , "mean_squared_logarithmic_error"]}

grid_search = GridSearchCV(estimator = regressor, param_grid = parametros, scoring= custom_scoring_function, cv=10)

grid_search = grid_search.fit(previsores, preco_real)
melhor_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_
