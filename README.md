## Circuncentro

Problemas de otimização convexa ocupam uma posição de grande importância em diversas áreas da matemática aplicada, ciência de dados, engenharia e economia, onde o objetivo é encontrar o mínimo de funções convexas sujeitas a restrições. Essa categoria de problemas possui boas propriedades que, em geral, permitem provar teoricamente convergência e eficiência dos métodos para sua solução. Entretanto, desafios persistem, especialmente em contextos de grande escala. Nos últimos anos, diversos trabalhos têm investigado o uso de geometria euclidiana refinada para aprimorar algoritmos de projeção e reflexão com foco no uso do circuncentro. Essa ideia tem sido empregada para acelerar métodos como os métodos de Douglas-Rachford, com resultados promissores. Neste projeto, propomos investigar o uso do circuncentro e suas propriedades geométricas para o desenvolvimento de novos algoritmos de otimização convexa, com ênfase na construção de esquemas iterativos que utilizam passos direcionados com base no circuncentro. Nosso objetivo é explorar essa abordagem para melhorar a taxa de convergência de métodos já estabelecidos e, adicionalmente, propor algoritmos capazes de encontrar o máximo de funções convexas. Pretendemos também estudar as propriedades teóricas desses algoritmos, incluindo sua convergência, e produzir implementações computacionais que possam ser testadas em problemas práticos de interesse.

Os algoritmos, por enquanto, foram todos implementados para resolver problemas de otimização linear, para facilitar a entrada aleatória de problemas e facilitar fazer testes numéricos. Porém, estamos estudando sua convergência para problemas de otimização convexa em geral.

Esse método busca implementar um método similar com o de gradiente de descida só que com a adição de restrições e de uma nova direção de busca. Para que o método mantenha se no conjunto viável, ao invés de fazer um passa na direção de menos o gradiente, utilizamos a direção "circuncentrica do gradiente", que é construida através do calculo do circuncentro dos vetores normais das restrições ativas e do gradiente.

##Algoritmos

criamos 3 algoritmos levemente diferentes com essa ideia de passo circuncentrico. Por enquanto, vamos chamar esses algortimos de Método de descida do Gradiente Circuncentrico.

#gradiente_circuncentrico.py
Este método possui um passo que leva sempre de um conjunto de restrições ativas para outro conjunto de restrições ativas, podendo extrair o máximo do uso do circuncentro.

#gradiente_circuncentrico_w.py
Este método escala o tamanho do passo feito dentro do conjunto viável para que possa se usar mais vezes a direção de máxima descida que é a direção do gradiente.

#gradiente_circuncentrico_deslize.py
Este método tenta impedir os algoritmos anteriores de fazer zig-zag entre dois conjuntos de restrições ativas diferentes. Fazendo  com que quando o método comece a repetir esses conjuntos a próxima iteração faz com que o ponto deslize no hiperplano de restrições ativas até ativar mais uma restrição

