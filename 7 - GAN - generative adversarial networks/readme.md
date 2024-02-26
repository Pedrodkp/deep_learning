# GAN - Generative Adversarial Networks

Sao redes que criam por elas mesmas  
- Em vez de dizer o que e um gato ou cachorro, ela vai aprender a criar um novo gato ou cachorro  
- Esta mais para uma imaginacao, tambem funciona para textos, audio, filmes e etc
- Pode ser utilizada tambem para aumento de resolucao, filtros visuais e etc  
- Podem ser utilizadas para conversao de texto para imagem  
- Traduçao de imagens para imagens, ex: faço um esboço ele cria a imagem, uma imagem de dia me retorna ela a noite e etc  

# Composicao
Possui um GERADOR (gera imagens) e um DISCRIMINADOR (analisa as imagens criadas pelo gerador para dizer se sao parecidas),
ambos precisam aprender sozinhos, logo sao duas redes neurais trabalhando juntas.  

## GERADOR
Inicializa com numeros aleatorios  
geralmente redes neurais DENSAS, mas podem ser CONVOLUCIONAIS tambem  
Deve possuir na camada de saida a quantidade de neuronios de acordo com a dimensao da imagem gerada  

## DISCRIMIADOR
Inicializa com as imagens de cachorro por exemplo 
geralmente rede classificao binaria DENSA mas pode ser a CONVOLUCIONAL, sempre com funcao sigmoid e back propagation  
Deve poussir na camada de saida apenas um neuronio, dizendo a probabilidade da imagem ser ou nao o que foi solicitado
  
# Processo

No começo o gerador vai gerar imagens sem sentido algum, e o discriminador vai facilmente desconsiderar mas vai adicionar como imagem invalida
para acelerar o processo, ou seja o discrimador e alimentado pelo gerador com imagens negativas e vai retornando a probabilidade das imagens serem
o que se pediu parar gera

# Install

keras not support GANs by default, so we need to install extra packages  
https://github.com/bstriner/keras-adversarial  
install:  
git clone https://github.com/bstriner/keras_adversarial.git  
cd keras_adversarial  
python setup.py install  