# Rotulação de Comentários sobre celulares no Mercado Livre
O objetivo desse projeto é obter um modelo capaz de ler feedbacks de compras de celulares realizadas no Mercado Livre e classificá-los em 3 rótulos: "Satisfeito", "Não atendeu as expectativas" e "Surgimento de um problema".
Para obter o modelo, foram testados três algoritmos para classificação: Naive Bayes, SVM (Support Vector Machine) e Random Forest e comparadas as curvas de aprendizagem, além de predições para melhorias na acurácia do modelo.
Para treinar o modelo, foram utilizados 6054 comentários obtidos de feedbacks de 27 aparelhos celulares do Mercado Livre rotulados em "Satisfeito", "Não atendeu as expectativas" e "Surgimento de um problema" por humanos.
# Pipeline
![Pipeline](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/e7c3bf42-9066-4995-9318-693373bceb22)
# Treinamento dos Modelos
## Leitura dos dados de treinamento
![lendo os dados](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/f79f504e-f7d8-42af-987d-123dd6e94c74)
## Limpeza e tratamento dos dados de treinamento
Para essa tarefa, o ferramental utilizado vieram das seguintes bibliotecas: Pandas, para a manipulação dos dados; NLTK (Natural Language Toolkit), para manipulação das palavras; Scikit-Learn, para implementação dos modelos; Spellchecker, para eventuais erros ortográficos.
### Tokenização, retirada de valores nulos e palavras com letras maiúsculas
A tokeniza é a função que recebe o dataframe com os dados de treinamento e transforma os comentários em vetores, fazendo com que cada palavra e sinal seja um elemento, dentro dela também aproveitei e implementei um comando pra retirada dos valores nulos e a transformação em qualquer letra maiúscula em minúscula, para uma palavra não ser interpretada mais de uma vez de forma diferente. Resumindo, essa função recebe o dataset com duas colunas: os comentários e os rótulos, e então cria uma terceira coluna para armazenar os comentários tokenizados, retornando um dataset com 3 colunas.

![tokeniza](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/909d7f09-8c39-4506-96c5-f2ff8d842e08)

Aqui ela trabalhando:

![tokeniza2](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/d4779e88-94a5-4a22-98b9-ff961bd3e44b)
### Limpeza
A função "limpa" retira tudo o que não for palavra dos comentários, números e todo tipo de sinalização

![func limpa](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/ae359517-475e-4b0e-a9f1-2c93cbc98e2f)
![palavras_limpas](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/9b69e89c-d843-4c40-8f86-8690be114e6c)
### Remoção de StopWords
A função removeStopWords retira as palavras menos importantes como os artigos e preposições. Contudo, por padrão, a palavra "não" também é retirada e eu julguei importante deixá-la, então implementei um comando pra ela continuar como uma palavra importante
![remove_stop_words](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/3bdbaba6-5a54-4295-8d3f-3fa089306e7f)
![remove_func](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/f8690969-3379-4e9b-84a2-9836425ba34d)
### Vetorização
Aqui é onde a mágica vai começar a acontecer, a função "vetoriza" transforma as palavras em valores numéricos e os coloca em uma matriz, a chamada Matriz Vetorada ou Matriz Esparsa, onde cada coluna estará relacionada com uma palavra e cada linha com sua ocorrência, o termo "Esparsa" vem do fato de que, como mesmo as palavras mais frequentes não ocorrem a todo momento, existem grandes espaços vazios nessa matriz. Não faz muito sentido visualizar ela aqui, por já se tratar de uma parte mais abstrata do processo, então vou colocar apenas a função:

![vetoriza](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/b8694004-cfaa-43f4-a088-fbc87d627e05)
![matriz vetorizada](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/8396c6bc-d0e0-4049-aa3d-ce944dd5626e)
### Codificação dos Rótulos
Agora que os comentários foram tranformados em números, chegou a hora dos rótulos. Aqui eles são convertidos assim: 0: "Não atendeu as expectativas", 1: "Satisfeito" e 2: "Surgimento de um Problema". Tudo vai ficar gravado em um Array.
![codifica rotulos](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/6f7a4124-2eb8-4692-83eb-baad8355b2e9)
![rotulos_codificados](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/87a95f91-1705-449f-9bb2-d598b500ba23)
