#PEGO DE:
#https://github.com/echen/restricted-boltzmann-machines
from rbm import RBM
import numpy as np

#OBJ recomendar filmes para o Leonardo

#6 filmes
#2 classificacoes
rbm = RBM(num_visible=6, num_hidden=3)

#valores para cada filme, 1 gostou, 0 se nao gostou, nao tem opcao nao ter visto
#na ordem: Ana, Marcos, Pedro, Claudia, Adriano, Janaina
n = 2 #0 nao gostou, 1, gostou, 2 nao assistiu, a lib nao aceita o None
base = np.array([[0,1,1,1,0,1],
                 [1,1,0,1,1,1],
                 [0,1,n,1,n,1],
                 [n,1,1,1,0,1], 
                 [1,1,0,1,0,1],
                 [1,1,n,1,1,1]])

filmes = ["Freddy x Jason", "O Ultimato Bourne", "Star Trek", "Exterminador do Futuro", "Norbit", "Star Wars"]

#treinamento com reconstrucao do no tentando encontrar neuronois especialistas
#max_epochs=5000, segundo o altor da biblioteca e o maior valor com algum aproveitamento, mais do que isso nao e muito util
rbm.train(base, max_epochs=5000)

##### VER #####
#primeira coluna e primeira linha sao quantidades de baias, logo ignoramos
#a partir da segunda linha temos os filmes, 6 linhas, ignoramos a primeira coluna e vemos os 2 neuronios
analise = rbm.weights
analise = analise[1:] #remove primeira linha
analise = np.delete(analise, 0, axis=1) #remove primeira coluna de cada linha
#assim fica mais facil enxergar os valores de cada neuronio
#maiores valores indicam maior especialidade, menor (ou negativos) signifcam maior distancia
print(analise)

##### TESTAR #####

leonardo = np.array([[n,1,n,1,0,n]])
rbm.run_visible(leonardo) #r: 1,1,1

##### RECOMENDACAO #####
camada_escondida = np.array([[1,1,1]])
recomendacao = rbm.run_hidden(camada_escondida) 

user = leonardo
recomendacao = recomendacao
for i in range(len(user[0])):
    #print(user[0,i])
    if user[0,i] == 0 and recomendacao[0,i] == 1:
        print(filmes[i])
