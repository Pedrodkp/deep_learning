# Redes neurais recorrentes
- Usadas para dados sequenciais (exemplo, prever o tempo)  
- Prever a proxima ação (prever o que uma pessoa vai fazer ou falar)  
- Muita utilizadas em processamento de linguagem natural (prever prox palavra em um texto)  
- Problemas de séries temporais (ex: preço na bolsa de valores)  

## Bulets
- Para entender o final precisa saber o que foi dito antes  
- Redes neurais tradicionais não armazenam informação no tempo (previsões independentes), esse tipo de rede possui loops que permitem que a informação persista (neoronio retorna a informação para ele mesmo)  
- Na pratica esse tipo de rede neural gera muitas cópias de si mesma  

## LSTM (Long-short term memory)
Para o exemplo de prever a palavra "céu" no exemplo: As nuvens estão no **céu**, é possível usar uma RNN simples.  
Porém, para o exemplo de preencher "eu falo português": Eu sou do Brasil ... (bastanta texto no meio) ... **eu falo português**, é dificil uma RNN lidar com isso, por isso usamos uma LSTM que se retro alimenta.  
LSTM é uma melhoria das redes anteriores, pois o neuronio ganha uma célula de memória (e diversos outros mecanismos).  

Conceitos:  
 - Forget gate> libera da memória
 - Input gate> adiciona na memória
 - Output gate> le da memória  

Processos (na sequência):  
 1) Decide o que será apagado (se o valor for zero, exemplo esta definindo o artigo de uma palavra, o artigo que esta vindo é masculino mas a palavra é feminino)  
 2) Decide o que será armazenado (verifica quais valores serão alterados, exemplo se achou varios objetos no texto ele vai criar vetores para verificar quais pronomes utilizar, exemplo apagou "o" e aponta para "a" para agua)
 3) Atualiza o estado efetivamente e propaga para as proximas execuções a memoria  
 4) Aplica a sigmoied e a tangente hiperbolica (filtros) e aplica a saida, IMPORTANTE: saida <> memoria  

 