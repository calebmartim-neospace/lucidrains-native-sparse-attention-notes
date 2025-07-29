## Visão geral
O repositório é uma implementação de Native Sparse Attention por Phil Wang (lucidrains), que é conhecido por suas implementações em PyTorch de pesquisa em IA. 
## Native Sparse Attention
Full Attention é computacionalmente caro para sequências muito longas pela complexidade $O(n^2)$ para construção da matriz $QKV$ e $O(n)$ para geração de cada token onde $n$ é o tamanho da sequência. Sparse attention tenta melhorar a performance ao ser seletivo com os tokens aos quais cada token atende.

Métodos de sparse attention até então consistiam em focar apenas em inferência. O time do DeepSeek propõe que isso limita as capacidades do modelo durante o seu treinamento ao treiná-lo com uma estratégia de full attention e jogá-lo num ambiente esparso durante seu uso. Hipoteticamente, o modelo poderia melhorar seus resultados ao se familiarizar com o seu ambiente esparso durante o seu treinamento.  

NSA propõe uma estratégia de sparse attention durante ambos treinamento para corrigir isso. Onde o modelo combina a técnica de compressão de tokens (visão geral) e seleção hierárquica (visão baixa) para o cálculo da sua atenção. 
## Introdução ao código
A classe principal é a `SparseAttention`. As três estratégias são aplicadas em paralelo.
### 1. `__init__.py` 
Simplesmente importa a classe `SparseAttention` do arquivo `native_sparse_attention/native_sparse_attention.py`
### 2. `tensor_typing.py`
Esse código é um wrapper redor de JAXTyping para habilitar dicas de tipagem em tensors do PyTorch. O wrapper deixa a sintaxe mais intuitiva no código. 

JAXTyping é uma biblioteca que habilita uma tipagem mais forte e especificação mais generalizável de formatos de tensors. Por exemplo:
```python
from jaxtyping import Float, Int, Bool

from torch import Tensor

# Determina as dimensões exatas do tensor

x: Float[Tensor, "batch=32 seq=128 dim=512"]

# Usa dimensões simbólicas (quaisquer tamanhos, mas devem ser os mesmos)

x: Float[Tensor, "batch seq dim"]

y: Float[Tensor, "batch seq dim"]
```

Com o wrapper, essa sintaxe fica mais simplificada:
```python
from tensor_typing import *

x: Float["batch seq dim"]

y: Float["batch seq dim"]
```
### 3. `compress_networks.py`
Uma coleção de redes neurais que podem ser usadas na branch de compressão de tokens. 
Lembrando, cada bloco de tokens é comprimido em um único valor a partir de alguma dessas redes neurais.

## Estratégias 
### Coarse-grained token compression
![[Pasted image 20250711134129.png]]
Os tokens são agrupados em uma sequência contígua e em uma janela de tamanho $n$ com um stride de tamanho $d$ comprime em um único só token os tokens vistos pela janela sequencialmente via um MLP. Isto faz com que o token atual obtenha a visão geral do seu contexto.
#### Implementação
Várias opções de compressão são dadas:
- ConvLinearCompress
- AttentionPool
- GroupedMLP
- SingleProjection
- CompressTransformer
Caso nenhuma seja especificada. Um simples MLP será usada. 

As sequência $K$ e $V$ são dividas primeiramente em blocos usando o encadeamento de alguns módulos:  
![[Pasted image 20250711133715.png]]
E cada bloco é comprimido com algumas opções de redes neurais. por padrão, uma MLP é usada:  
![[Pasted image 20250711134943.png]]
### Fine-grained token selection
Após a compressão, os top-$k$ blocos comprimidos com maiores pontos de compressão são descomprimidos e cada token que se encontrava neste bloco tem sua atenção calculada com o token atual. Isso faz com que o token atual obtenha informações à baixo nível de tokens que são relevantes para a sua contextualização  
```mostrar como o cara faz```
### Sliding window
Uma janela de tamanho $w$ é mantida onde attention é calculado com os últimos tokens. Isso é feito para o token ter boas noções do seu contexto local.
![[Pasted image 20250711130753.png]]
O código usa a biblioteca `local_attention`.  
