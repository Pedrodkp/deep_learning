#PEGO DE:
#https://github.com/echen/restricted-boltzmann-machines
from rbm import RBM
import numpy as np

#6 filmes
#2 classificacoes
rbm = RBM(num_visible=6, num_hidden=2)

#valores para cada filme, 1 gostou, 0 se nao gostou, nao tem opcao nao ter visto
#os 3 primeiros seriam de terror, os 3 ultimos de comedia
base = np.array([[1,1,1,0,0,0],
                 [1,0,1,0,0,0],
                 [1,1,1,0,0,0],
                 [0,0,1,1,1,1], 
                 [0,0,1,1,0,1],
                 [0,0,1,1,0,1]])

filmes = ["A bruxa", "Invocaçao do mal", "O chamado",\
          "Se beber nao case", "Gente grande", "American pie"]

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
#primeiro neuronio e comedia, segundo neuronio e de terror, ele determina a ordem sozinho
#mas e possivel saber pelo print da analise

new_user1 = np.array([[1,1,0,1,0,0]])
#primeiro neuronio e comedia, segundo neuronio e de terror 
rbm.run_visible(new_user1) #r: 0,1 logo terror

new_user2 = np.array([[0,0,0,1,1,0]])
rbm.run_visible(new_user2) #r: 1,0 logo comedia

##### RECOMENDACAO #####
camada_escondida = np.array([[0,1]])
recomendacao1 = rbm.run_hidden(camada_escondida) #mostra quais neuronois recomenda, basta ignorar os que ele ja viu e recomendar os outros

camada_escondida = np.array([[1,0]])
recomendacao2 = rbm.run_hidden(camada_escondida) #recomendou tbm um filme de comedia: 0,0,1,1,1,1

user = new_user2
recomendacao = recomendacao2
for i in range(len(user[0])):
    #print(user[0,i])
    if user[0,i] == 0 and recomendacao[0,i] == 1:
        print(filmes[i])
