#RBM - Restricted Boltzmann Machines

E do tipo aprendizagem nao supervisionada.  
Todos os neuronios ligam com todos nesse tipo de rede (inclusive os de entrada, intermediarios e saida).  
Utilizada para reduçao de dimensionalidade para machine learning (reduzindo tempo de aprendizado e processamento)
e tambem pode ser utilizada para recomendaçoes.  
Esse tipo de rede nao possui "camada de saida", as entradas tbm sao alteradas, e esse tipo de rede tem por objetivo reajustar
o valor de todos os neuronios.  
Tambem pode ser utiliada para monitorar um sistema e detecçao de outliers (anomalias), exemplo ele aprende como funciona uma maquina
com todos os seus parametros, aprendendo o padrao se ela sair do padrao ele vai conseguir detectar.  

## Quantos mais nos, mais conexoes exponencioalmente, logo fica inviavel rapido.  

Por isso mudamos um pouco o formato da 'Boltzmann Machines Classic' criando a 'Restricted Boltzmann Machines', que tem um desenho um pouco
mais tradicional, com todos ligando em todos da entrada para a oculta, mas nao permite que neuronois da camada oculta tenham ligacoes entre si.

## Constrative divergence (aprendizagem)

Esse tipo de aprendizagem e parecido com as outras redes, porem apos calcular, ele vai reconstruir os registros.  
A execuçao termina apenas quando os valores sao os mesmos ou a numero de epocas e atingindo, e os pesos so sao atualizados
no final.  

### Deep Belief Network (DBN)

Acrescenta muitas camadas intermediarias (escondidas), utilizadas para sistemas muito complexos e com muitos dados.  

#pydbm

Biblioteca robusta em Python que você pode construir: Restricted Boltzmann Machine (RBM), Deep Boltzmann Machine (DBM), Long Short-Term Memory Recurrent Temporal Restricted Boltzmann Machine (LSTM-RTRBM) e Shape Boltzmann Machine (Shape-BM
