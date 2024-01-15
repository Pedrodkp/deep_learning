import numpy as np

###
### funcoes de ativacoes sao necessarias para determinar a saida de um neoronio
###

#(transfer function) muito simples quase nao utilizada, problemas linearmente separaveis
#utiliza para problemas AND e OR, nao funciona para problemas XOR
def step(soma):
    if (soma >= 1):
        return 1
    return 0

print(f"step : {step(0.358)}")
print(f"step : {step(-0.358)}")

#(funcao sigmoid) utilizada para problemas binarios, quando tem apenas duas classes
#nao aceita negativo, vai de 0 a 1
def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

print(f"sigmoid : {sigmoid(0.358)}")
print(f"sigmoid : {sigmoid(-0.358)}")
print(f"sigmoid : {sigmoid(2.1)}")

#(hyperbolic tanget) utilizada para problemas binarios, quando tem apenas duas classes 
#mas permite entradas negativas fortes e zero sera neutro, vai de -1 a 0 a 1
def tahn(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))

print(f"tahn : {tahn(0.358)}")
print(f"tahn : {tahn(-0.358)}")
print(f"tahn : {tahn(2.1)}")

#(relu) nao tem valor maximo, utilizada em redes neurais convulocionais, para visao computacional ou muitas camadas
#funcao mais utilizada
def relu(soma):
    if (soma >= 0):
        return soma
    return 0

print(f"relu : {relu(0.358)}")
print(f"relu : {relu(-0.358)}")

#(linear) retorna valor passado, nao faz nada
#muito utilizada em problemas de regressao, para por exemplo prever prsecos ja que o valor nao pode ser mudado
#ex: prever o valor de um carro
def linear(soma):
    return soma

print(f"linear : {linear(0.358)}")
print(f"linear : {linear(-0.358)}")

#(softmax) uma das mais importantes, utilizadas para retornar probabilidades com problemas de mais de duas classes
#exemplo: prever uma cor, entre verde, azul, roxo e etc ou seja, mais de duas opcoes possiveis de saida
#x - vetor a receber
#ex - exponencial
def softmax(x):
    ex = np.exp(x)
    return ex / ex.sum()

print(f"softmax : {softmax([5,2,1.3])}")