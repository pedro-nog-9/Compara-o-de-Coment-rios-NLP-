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
Para essa tarefa, o ferramental utilizado vieram das seguintes bibliotecas: Pandas, para a manipulação dos dados; NLTK (Natural Language Toolkit), para manipulação das palavras; Scikit-Learn, para implementação dos modelos; Spellchecker, para eventuais erros ortográficos; joblib, para salvar e abrir os modelos.
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
## Implementando os modelos
### Naive Bayes
Essa função recebe a matriz esparsa e o array com os números dos rótulos e retorna o modelo pronto. Foi separado 75% dos dados para treinamento e 25% para teste, o hiperparâmetro alpha foi definido em 0.1 e também foi implementado um print para mostrar a acurácia calculada nos dados de teste. Vale lembrar que essa acurácia geralmente é maior que quando o modelo é aplicado na realidade.

![modeladoraNB](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/109b5af9-f62b-423f-a9ea-fb202b214838)

![implementaNB](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/d1a49914-d616-4ae3-b903-8d36bbe01d33)

### SVM (Support Vector Machine)

Na SVM, foi usado como hiperparâmetros kernel='rbf', C=1, gamma='scale', degree=1. Considerando a mesma proporção de teste/treinamento e acurácia

![modeladoraSVM](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/080e732e-8601-4c4f-a508-14f247414a8d)

![svm implementada](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/b8a4c36c-abfa-4811-869d-8e69d85b0896)

### Random Forest
A Random Forest aqui foi implementada com hiperparâmetros default, pois nenhum valor diferente relevante foi encontrado.

![random forest](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/4af065fb-a0fe-4622-a442-9c83a1bb900a)

![random forest imp](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/dfbb1e95-9945-4297-8df7-79308f0394a8)

### Salvando os modelos

![salvando modelos](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/6196470d-7d99-4a65-9215-5f81a5b4f836)

É necessário saber a dimensão da matriz esparsa, mais especificamente quantas features (colunas) ela possui. Isso acontece porque na aplicação real o modelo espera receber uma matriz esparsa exatamente do mesmo tamanho.

![matrizesparsa](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/fd4f8b7a-64ae-41c8-b6ec-1038412dfd06)

A matriz tem 6272 features.

# Coleta de dados reais

## Extração Direta

Aqui é feito um Web Scrapping a partir de um link dos comentários de um modelo específico, no caso, um Motorola Edge 30. Para essa tarefa as bibliotecas utilizadas foram a Beautiful Soup, Selenium, Requests para a raspagem dos dados e Time para manipulação do tempo. E aqui um ponto importante: A biblioteca Selenium foi necessária pois a página dos comentários é dinâmica e o código só fica disponível quando é aberto pelo navegador, então a Selenium faz esse papel de simular um humano acessando o site por um navegador. Além disso, para carregar todos os comentários é necessário que a página seja toda rolada para baixo e o servidor do Mercado Livre percebe se uma máquina fizer isso e corta a conexão, logo, para ser possível o carregamento de todos os comentários a página precisa ser rolada com o scroll do mouse. Após o código ser gerado pelo navegador, induzido pela Selenium, a Beautiful Soup recebe o código e faz o trabalho normalmente.

### Abrindo o navegador com a Selenium

![abrindo navegador](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/cef9b004-00af-4dbe-af93-fc4bf4010613)

Aqui o navegador abre automaticamente e é só rolar a página para baixo

![abrindo navegador 2](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/e5375b27-1bdd-4938-b639-a5cce052bb7b)

### Passando a página para a Beautiful Soup depois de descer a página toda para baixo

![expansora](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/ae4046d4-00fd-44a4-afbc-9078c59d28a6)

![expansora2](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/391860d1-039b-4818-b66c-2f97bf09f34b)

### Extraindo apenas os comentários do código HMLT que foi recebido

![extratoraBS](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/05d2698f-b663-4594-b651-2d12c2f69b05)

![extratoraBS2](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/0469bf1a-5051-4df6-8b08-21a73cf985bb)

### Salvando os novos comentários na pasta

![salvar](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/eeb33b37-a1ee-44ae-9c2f-0d779bbd2f7a)

![salvar2](https://github.com/pedro-nog-9/Rotulacao_de_Comentarios/assets/127139232/dd51f303-5a1b-4883-9541-a269c05da672)

# Aplicação dos novos dados nos modelos
## Limpeza e tratamento dos novos dados
