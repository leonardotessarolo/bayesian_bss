# Estimação Pontual Bayesiana via critério Minimum Mean Square Error (MMSE)

#### De Paulino, Turkman e Murteira (2003): estimativa $\hat{B}_{MMSE}$ é obtida como sendo a média (valor esperado) da distribuição a posteriori. Considerando $\mathcal{B}$ como sendo o conjunto de todas as matrizes B possíveis:

$$\hat{B}_{MMSE}=E[B]=\int_{\mathcal{B}}B \cdot P(B|X) dB$$ 


#### Demonstra-se, de maneira simples, que este critério coincide com a minimização do erro quadrático médio, sendo portanto chamado de MMSE. Considera-se que o erro quadrático médio para uma estimativa qualquer $\hat{B}$ seja dada por:

$$ MSE = \int_{\mathcal{B}} (\hat{B}-B)^{2} \cdot P(B|X) dB $$

#### 
# Simulação de Monte Carlo

#### De Brémaud (2020): quando não possível ou muito complexa em termos de resolução analítica, a integral para estimação de $\hat{B}_{MMSE}$ pode ser aproximada gerando-se amostras $B_{i}$ de B segundo a distribuição $P(B|X)$, e computando a média destas amostras. A Lei Forte dos Grandes Números garante a convergência.

$$\hat{B}_{MMSE}=E[B]=\int_{\mathcal{B}}B \cdot P(B|X) dB= \lim_{N \to \infty} \sum_{i=1}^{N}B_{i}$$

#### Há, ainda, o desafio de se obter amostras $B_{i}$ distribuídas de acordo com $P(B|X)$. Para este fim, utilizam-se os métodos MCMC.

#### 

# Métodos de Monte Carlo via Cadeia de Markov (Markov Chain Monte Carlo - MCMC)

#### De Paulino, Turkman e Murteira (2003): 

#### "A ideia básica por detrás desses métodos é a da transformar o problema estático em consideração num problema de natureza dinâmica, construindo para o efeito um processo estocástico temporal, artificial, que seja fácil de simular, e que convirja, fracamente, para a distribuição original. Este processo temporal é, em geral, uma cadeia de Markov homogênea cuja distribuição de equilíbrio é a distribuição que se pretende simular. Para implementar este método há necessidade de saber construir cadeias de Markov com distribuições de equilíbrio específicas."

#### 

# Algoritmo Metropolis-Hastings

#### De Mira (2005): no caso de se desejar obter amostras da distribuição a posteriori $P(B|X)$, o algoritmo Metropolis-Hastings (Metropolis et al., 1953; Hastings, 1970) se constitui pela aplicação sequencial da seguinte regra:



#### 1. Dado B_{i}, o valor de B num instante qualquer i, uma movimentação para B_{i+1} é proposta segundo uma distribuição $Q(B_{i},B_{i+1})$. Por exemplo, Q pode ser uma distribuição normal centrada em $B_{i}$, e com variância pré-determinada.

#### 2. O valor proposto de $B_{i+1}$ será aceito com probabilidade $\alpha(B_{i}, B_{i+1})$, onde:

$$ \alpha(B_{i}, B_{i+1})= \min \left[ \; 1, \; \frac{P(B_{i+1}|X) \cdot Q(B_{i},B_{i+1})}{P(B_{i}|X) \cdot Q(B_{i},B_{i+1})} \; \right] $$

#### 3. Caso não seja aceito o novo valor, faz-se $B_{i+1}=B_{i}$.

#### 

#### Obs: caso a função Q seja simétrica, como por exemplo uma distribuição normal, tem-se que $Q(B_{i},B_{i+1})=Q(B_{i+1},B_{i})$. Neste caso, a expressão da probabilidade de aceite se reduz a:

$$ \alpha(B_{i}, B_{i+1})= \min \left[ \; 1, \; \frac{P(B_{i+1}|X)}{P(B_{i}|X)} \; \right] $$

#### Este é o algoritmo originalmente proposto em Metropolis et al. (1953), com a generalização para funções Q não simétricas tendo sido proposta em Hastings, 1970. Neste ensaio, utiliza-se Q normal, e portanto simétrica.

#### 

# Cálculo de $\alpha(B_{i}, B_{i+1})$

#### Uma vez que as probabilidades a posteriori envolvidas no cálculo de $\alpha(B_{i}, B_{i+1})$ são muito pequenas, faz-se uma adaptação das equações envolvidas para que estejam expressas em termos das log-posterioris:

$$ \log \alpha(B_{i}, B_{i+1})= \min \left[ 0, \; \log P(B_{i+1}|X) - \log P(B_{i}|X) \; \right] $$

Portanto, tem-se:

$$ \alpha(B_{i}, B_{i+1})= \exp \left( \min \left[ 0, \; \log P(B_{i+1}|X) - \log P(B_{i}|X) \; \right] \right)$$

#### Utiliza-se a equação da log-posteriori de Djafari (2000):
$$ \log P(\boldsymbol{B}|y) = T \log |det({B})| + \sum_{t} \sum_{i} \log p_{i}(y_{i}(t)) + \log P(\boldsymbol{B}) + cte$$


