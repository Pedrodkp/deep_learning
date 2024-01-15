#start conda

source /home/upiara/anaconda3/bin/activate
conda init

#start conda navigator

anaconda-navigator

#requirements

pip install tensorflow
pip install keras
pip install scikeras[tensorflow]      # gpu compute platform
pip install scikeras[tensorflow-cpu]  # cpu 

#resumo de termos

**Pesos (Weights):**
- Na context de redes neurais, os pesos são parâmetros ajustáveis que a rede aprende durante o treinamento.
- Cada conexão entre neurônios possui um peso que determina a importância dessa conexão para a saída da rede.

**Épocas (Epochs):**
- Uma época representa uma passagem completa por todo o conjunto de treinamento durante o processo de treinamento de uma rede neural.
- O número de épocas indica quantas vezes a rede neural foi treinada usando todos os exemplos disponíveis.

**Delta:**
- Delta geralmente se refere à mudança ou variação em alguma quantidade.
- No contexto do treinamento de redes neurais, o termo "delta" muitas vezes é usado para representar a mudança nos pesos durante a atualização.

**Função degrau (Step Function):**
- Uma função de ativação utilizada em perceptrons e em redes neurais de uma camada.
- A função degrau produz uma saída binária (0 ou 1) com base em um limiar. Se a entrada ultrapassa o limiar, a saída é 1; caso contrário, é 0.

**Backpropagation:**
- Um algoritmo de treinamento utilizado em redes neurais para ajustar os pesos com base no gradiente da função de perda.
- Envolve a propagação do erro da saída para as camadas anteriores, permitindo a atualização iterativa dos pesos para minimizar a perda.

**Descida do Gradiente Estocástico (Stochastic Gradient Descent - SGD):**
- Um algoritmo de otimização usado para treinar redes neurais.
- Os pesos são atualizados após cada exemplo de treinamento, tornando o processo mais rápido e adaptável.

