# Libraries
import torch
import sublayers

class EncoderBlock(torch.nn.Module):
    '''
    Block composing an encoder.

    Args:
        d_model (int): Dimension of hidden states.
        d_k (int): Dimension of the linear projection for Q and K.
        d_v (int): Dimension of the linear projection for V.
        h (int): Number of heads.
        d_ff (int): Dimension of inner-layer.

    Attributes:
        mha (torch.nn.Module): Multi-Head Attention sublayer.
        ffn (torch.nn.Module): Position-wise Feed-Forward Network.
        dropout_(1/2) (torch.nn.Dropout): Dropout modules.
        layernorm_(1/2) (torch.nn.LayerNorm): Layer normalization modules.
    '''
    def __init__(self, d_model: int, d_k: int, d_v: int, h: int, d_ff: int):
        super(EncoderBlock, self).__init__()
        self.mha = sublayers.MHA(d_model, d_k, d_v, h)
        self.ffn = sublayers.FFN(d_model, d_ff)
        self.dropout_1 = torch.nn.Dropout(0.1)
        self.dropout_2 = torch.nn.Dropout(0.1)
        self.layernorm_1 = torch.nn.LayerNorm(d_model)
        self.layernorm_2 = torch.nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        '''
        Forward hidden states through an encoder block.

        Args:
            x (torch.Tensor): Hidden states of dimension (batch_size, max_seq_length, d_model).
            attention_mask (torch.Tensor): Mask for padding.

        Returns:
            res (torch.Tensor): Output of the block of dimension (batch_size, max_seq_length, d_model).
        '''
        x_int = self.mha.forward(x, x, x, attention_mask)
        x_int = self.dropout_1(x_int)
        x_int = torch.add(x_int, x)
        x_int = self.layernorm_1(x_int)

        res = self.ffn.forward(x_int)
        res = self.dropout_2(res)
        res = torch.add(res, x_int)
        res = self.layernorm_2(res)
        return res

class Encoder(torch.nn.Module):
    '''
    Encoder part (without the embedding) of a transformer.

    Args:
        n_layers (int): Number of blocks for the encoder.
        d_model (int): Dimension of hidden states.
        d_k (int): Dimension of the linear projection for Q and K.
        d_v (int): Dimension of the linear projection for V.
        h (int): Number of heads.
        d_ff (int): Dimension of inner-layer.

    Attributes:
        blocks (torch.nn.ModuleList): List of encoder blocks.
    '''
    def __init__(self, n_layers: int, d_model: int, d_k: int, d_v: int, h: int, d_ff: int):
        super(Encoder, self).__init__()
        self.blocks = torch.nn.ModuleList([EncoderBlock(d_model, d_k, d_v, h, d_ff) for _ in range(n_layers)])
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        '''
        Forward inputs embeddings through the encoder.

        Args:
            x (torch.Tensor): Hidden states of dimension (batch_size, max_seq_length, d_model).
            attention_mask (torch.Tensor): Mask for padding.

        Returns:
            res (torch.Tensor): Output of the encoder of dimension (batch_size, max_seq_length, d_model).
        '''
        for block in self.blocks:
            x = block.forward(x, attention_mask)
        return x