"""
@misc{labml,
 author = {Varuna Jayasiri, Nipun Wijerathne},
 title = {labml.ai Annotated Paper Implementations},
 year = {2020},
 url = {https://nn.labml.ai/},
}
"""

import torch
from torch import nn
import math
from typing import Optional, List, Union, Tuple
import pytorch_lightning as pl

class PrepareForMultiHeadAttention(nn.Module):
    """
    <a id="PrepareMHA"></a>

    ## Prepare for multi-head attention

    This module does a linear transformation and splits the vector into given
    number of heads for multi-head attention.
    This is used to transform **key**, **query**, and **value** vectors.

    Finally, it reshapes the tensor to `(seq_len, batch_size, heads, d_k)`.
    """

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        head_shape = x.shape[:-1]
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)
        return x


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int = 10_000):
        """
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()

        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        """
        Cache $\cos$ and $\sin$ values
        """
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return

        seq_len = x.shape[0]

        theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)

        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)

        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # Cache them
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x: torch.Tensor):
        d_2 = self.d // 2

        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        self._build_cache(x)
        x_rope, x_pass = x[..., :self.d], x[..., self.d:]

        neg_half_x = self._neg_half(x_rope)
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])

        return torch.cat((x_rope, x_pass), dim=-1)


class RotaryPEMultiHeadAttention(pl.LightningModule):
    def __init__(self, 
                 d_model: int, 
                 heads: int, 
                 dropout_prob: float, 
                 rope_percentage: float = 0.5, 
                 bias: bool = False):
        super().__init__()

        self.d_k = d_model // heads
        self.heads = heads

        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)

        self.softmax = nn.Softmax(dim=1)

        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)

        self.attn = None

        d_rope = int(self.d_k * rope_percentage)
        self.query_rotary_pe = RotaryPositionalEmbeddings(d_rope)
        self.key_rotary_pe = RotaryPositionalEmbeddings(d_rope)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        return torch.einsum('ibhd,jbhd->ijbh', self.query_rotary_pe(query), self.key_rotary_pe(key))

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor]]] = None):
        
        #Permute dimensions 0 and 1 for qkv
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        seq_len, batch_size, _ = query.shape

        if mask is not None:
            if isinstance(mask, tuple):
                mask_q = mask[0].unsqueeze(2)
                mask_k = mask[1].unsqueeze(1)

            else:
                mask_q = mask.unsqueeze(2)
                mask_k = mask.unsqueeze(1)

            mask = mask_q * mask_k
            mask = mask.permute(1, 2, 0).unsqueeze(-1)

            # #print the first batch of mask, mask_q, mask_k to verify the shapes
            # if torch.sum(mask_q[0, :, 0]) != torch.sum(mask_k[0, 0, :]):
            #     print(mask[:, :, 0].squeeze())
            #     print(mask_q[0, :, 0].squeeze())
            #     print(mask_k[0, 0, :].squeeze())
            #     print(torch.sum(mask_q[0, :, 0]), torch.sum(mask_k[0, 0, :]), torch.sum(mask[:, :, 0]))
            #     exit()

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        scores = self.get_scores(query, key)

        scores *= self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e38)

        attn = self.softmax(scores)

        attn = self.dropout(attn)

        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        self.attn = attn.detach()

        x = x.reshape(seq_len, batch_size, -1)

        return self.output(x).permute(1, 0, 2)
    
    
if __name__ == '__main__':
    # Create an instance of the model
    model = RotaryPEMultiHeadAttention(d_model=512, 
                                        heads=8, 
                                        dropout_prob=0.1, 
                                        rope_percentage=0.5, 
                                        bias=True)

    # Create some random data
    query = torch.randn(32, 100, 512)
    key = torch.randn(32, 256, 512)
    value = torch.randn(32, 256, 512)

    mask1 = torch.randn(32, 100)
    mask2 = torch.randn(32, 256)
    # mask = None

    # Pass the data through the model
    output = model(query, key, value, mask=(mask1, mask2))
    print(output.shape)

