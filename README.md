## Annotating the Annotated Transformer

This repo contains a simple demonstration of pytorch implementation of the transformer architecture. A more detailed post is [Annotating the Annotated Transformer](https://uriyeobi.github.io/2024-01-15/annot-annot-transformer), originally inspired by [Annotated Tranformer](https://nlp.seas.harvard.edu/annotated-transformer/).

The dimensions used in the multi-head attention are as follows:

```python
class MultiHeadedAttention(nn.Module):

    def __init__(self, d_in, D_q, D_k, D_v, d_out, h, dropout):
        super(MultiHeadedAttention, self).__init__()

        self.linear_Q = nn.Linear(d_in, D_q, bias=False)
        self.linear_K = nn.Linear(d_in, D_k, bias=False)
        self.linear_V = nn.Linear(d_in, D_v, bias=False)
        self.linear_Wo = nn.Linear(D_v, d_out)
        
        {...}
    
    def forward(self, query, key, value, mask=None):
        {...}
```

For example:


<img src="https://github.com/uriyeobi/annotated_transformer/blob/main/annotated_transformer/attn_dimensions.png?raw=true" width="800rem">

