# An Implementation of Transformer in Translation from English to Chinese


Reference:

>Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[J]. Advances in neural information processing systems, 2017, 30.
>
>https://arxiv.org/pdf/1706.03762.pdf
>
>https://zhuanlan.zhihu.com/p/144825330

Steps from 0 to 16 in [transformer.ipynb](transformer.ipynb)(for data_small) & [transformer_L.ipynb](transformer_L.ipynb)(for data_large: translation2019zh) are following:

## 0.Import some dependences & set some parameters

## 1.Data Preparation(tokenize, word2id, add padding & mask, batchnize)

- 1).data_small

    baidudisk link: https://pan.baidu.com/s/1W16jPrdrP-uvzv4ZqYIvlw?pwd=n5ka code：n5ka

- 2).data_large

    google drive link: https://drive.google.com/u/0/uc?id=1EX8eE5YWBxCaohBO8Fh4e2j3b9C2bTVQ&export=download

    baidudisk link: https://pan.baidu.com/s/1KMHiroCl9wq8E4pRO_2oqw?pwd=6qts code：6qts

## 2.Input Embedding

$$InputEmbedding(x) = Embedding(x) * \sqrt {d_{model}}$$

## 3.Positional Encoding

$$PE_{(pos, 2i)} = \sin (\frac{pos}{1000^{\frac{2i}{d_{model}}}})$$
$$PE_{(pos, 2i+1)} = \cos (\frac{pos}{1000^{\frac{2i}{d_{model}}}})$$

## 4.Multi-Head Attention

$$
\begin{gather*}
    MHA(X) = Concatenate(Attention_i(X)) * W_C\\
    i \in [1, numheads]\\
	Attention(X) = SelfAttentionOrContextAttention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V\\
	Q = Linear(X) = X * W_Q\\
	K = Linear(X) = X * W_K\\
	V = Linear(X) = X * W_V\\
	d_k = d_{model} // numheads\\
	W_C, W_Q, W_K, W_V = clones(nn.Linear(d_{model}, d_{model}), 4)
\end{gather*}
$$

Above 3 X during the calculating Q, K, V will be same if SelfAttention else different.

## 5.LayerNorm

$$LayerNorm(x) = \alpha * \frac{x_{ij} - \mu_{i}}{\sqrt{\sigma _i^2 + \epsilon}} + \beta$$

## 6.Positionwise FeedForward

$$PositionwiseFeedForward(X_{attn}) = Linear(Activate(Linear(X_{attn})))$$

## 7.Utilities class: SublayerConnection & clones

$$SublayerConnection(X) = X + SubLayer(X)$$
$$clones(X, N) = [X repeat N times]$$

## 8.EncoderLayer & Encoder(N_head EncoderLayers)

## 9.DecoderLayer & Decoder(N_head DecoderLayers)

## 10.Transformer

<img src='https://pic1.zhimg.com/80/v2-4b53b731a961ee467928619d14a5fd44_720w.jpg' text-align='center'/>

### 10.1.Encoder

- 1).InputEmbedding + PositionalEncoding

$$X_{emb} = InputEmbedding(X) + PositionalEncoding(pos)$$

- 2).MultiHeadSelfAttention

$$
\begin{gather*}
    X_{attn} = MHA(X_{emb}) = Concatenate(Attention_i(X_{emb})) * W_C\\
    i \in [1, numheads]\\
    Attention(X_{emb}) = SelfAttention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V\\
    Q = Linear(X_{emb}) = X_{emb} * W_Q\\
    K = Linear(X_{emb}) = X_{emb} * W_K\\
    V = Linear(X_{emb}) = X_{emb} * W_V\\
    d_k = d_{model} // numheads\\
    W_C, W_Q, W_K, W_V = clones(nn.Linear(d_{model}, d_{model}), 4)
\end{gather*}
$$

- 3).SublayerConnection + Norm

$$X_{attn} = LayerNorm(X_{attn})$$
$$X_{attn} = X + X_{attn}$$

- 4).PositionwiseFeedForward

$$X_{hidden} = Linear(Activate(Linear(X_{attn})))$$

- 5).Repeat 3)

$$X_{hidden} = LayerNorm(X_{hidden})$$
$$X_{hidden} = X_{attn} + X_{hidden}$$

- 6).Repeat 2) ~ 5) * N 

    Let the output of previous 5) be the input of next 2), repeating N times.

### 10.2.Decoder

- 1).InputEmbedding + PositionalEncoding

$$Y_{emb} = InputEmbedding(Y) + PositionalEncoding(pos)$$

- 2).MultiHeadSelfAttention

$$
\begin{gather*}
    Y_{attn1} = MaskedMHA(Y_{emb}) = Concatenate(Attention_i(Y_{emb})) * W_C\\
    i \in [1, numheads]\\
    Attention(Y_{emb}) = SelfAttention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V\\
    Q = Linear(Y_{emb}) = Y_{emb} * W_Q\\
    K = Linear(Y_{emb}) = Y_{emb} * W_K\\
    V = Linear(Y_{emb}) = Y_{emb} * W_V\\
    d_k = d_{model} // numheads\\
    W_C, W_Q, W_K, W_V = clones(nn.Linear(d_{model}, d_{model}), 4)
\end{gather*}
$$

- 3).SublayerConnection + Norm

$$Y_{attn1} = LayerNorm(Y_{attn1})$$
$$Y_{attn1} = Y + Y_{attn1}$$

- 4).MultiHeadContextAttention

$$
\begin{gather*}
    Y_{attn2} = MHA(Y_{attn1}, M, M) = Concatenate(Attention_i(Y_{attn1}, M, M)) * W_C\\
    i \in [1, numheads], M = X_{hidden}\\
    Attention(Y_{attn1}, M, M) = ContextAttention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V\\
    Q = Linear(Y_{attn1}) = Y_{attn1} * W_Q\\
    K = Linear(M) = M * W_K = X_{hidden} * W_K\\
    V = Linear(M) = M * W_V = X_{hidden} * W_V\\
    d_k = d_{model} // numheads\\
    W_C, W_Q, W_K, W_V = clones(nn.Linear(d_{model}, d_{model}), 4)
\end{gather*}
$$

- 5).Repeat 3)

$$Y_{attn2} = LayerNorm(Y_{attn2})$$
$$Y_{attn2} = Y_{attn1} + Y_{attn2}$$

- 6).PositionwiseFeedForward

$$Y_{hidden} = Linear(Activate(Linear(Y_{attn2})))$$

- 7).Repeat 3)

$$Y_{hidden} = LayerNorm(Y_{hidden})$$
$$Y_{hidden} = Y_{attn2} + Y_{hidden}$$

- 8).Repeat 2) ~ 7) * N 

    Let the output of previous 7) be the input of next 2), repeating N times.

### 10.3.linear + log_softmax

## 11.Make a real Transformer model

## 12.Smooth the label(implement by KLdivloss)

## 13.Compute the loss

## 14.Set optimizer with a warmupdown learning rate

$$lr = lr_{base} * [d_{model}^{-.5} * \min{(step\_ num^{-.5}, step\_ num*warmup\_  steps^{-1.5})}]$$

The lr increases linearly with a fixed warmup_steps, decreases proportional to the inverse square root of step_num when it reached warmup_steps(here is 4000).

The base optimizer is Adam with beta1=.9, beta2=.98, epsilon=1e-9.

## 15.Train and Validation

## 16.Prediction or say Translation