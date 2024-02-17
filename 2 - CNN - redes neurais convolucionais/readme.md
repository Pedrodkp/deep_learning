#Etapas

Para ter a visão computacional, precisamos passar por algumas etapas para ter as caracteristicas principais. Isto para reduzir a matriz que representa a imagem.

1 - Operador de convolução (filtros e kernels), multiplicação dos pixels por regiões
(detecão de bordas, suavização, realce de caracteristicas e extração)
2 - Pooling. extração do valor máximo para pegar o pixel mais destacado, o objetivo é reduzir a carga de trabalho
3 - Flatteing. Transforma a saída tridimensional em convolucional, a partir daqui, se torna uma rede neural normal.
ex: se a saída de uma camada convolucional ou de pooling tem as dimensões (altura, largura, profundidade), a operação de flattening transformará isso em uma única dimensão de tamanho (altura * largura * profundidade).

#Simulador

https://adamharley.com/
https://adamharley.com/nn_vis/
https://adamharley.com/nn_vis/cnn/3d.html
https://pianalytix.com/cnn-convolutional-neural-network/

#Explicacao de teoria
Como sabemos a rede aprende a determinar os pesos, na explicacao da rede neural densa isso fica bem ilustrado do por que o peso determina a resposta.

#CNN vs SVM
Antes usavamos SVM, mas devido a aprimoramento o CNN tem se comportado melhor