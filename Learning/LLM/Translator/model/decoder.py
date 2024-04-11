# Libraries
import torch
import sublayers

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

class DecoderBlock(torch.nn.Module):
    '''
    Block composing a decoder.

    Args:
        d_model (int): Dimension of hidden states.
        d_k (int): Dimension of the linear projection for Q and K.
        d_v (int): Dimension of the linear projection for V.
        h (int): Number of heads.
        d_ff (int): Dimension of inner-layer.

    Attributes:
        masked_mha (torch.nn.Module): Masked Multi-Head Attention.
        mha (torch.nn.Module): Multi-Head Attention.
        ffn (torch.nn.Module): Position-wise Feed-Forward Network.
        dropout_(1/2/3) (torch.nn.Dropout): Dropout modules.
        layernorm_(1/2/3) (torch.nn.LayerNorm): Layer normalization modules.
    '''
    def __init__(self, d_model: int, d_k: int, d_v: int, h: int, d_ff: int):
        super(DecoderBlock, self).__init__()
        self.n_heads = h
        self.masked_mha = sublayers.MHA(d_model, d_k, d_v, h)
        self.mha = sublayers.MHA(d_model, d_k, d_v, h)
        self.ffn = sublayers.FFN(d_model, d_ff)
        self.dropout_1 = torch.nn.Dropout(0.1)
        self.dropout_2 = torch.nn.Dropout(0.1)
        self.dropout_3 = torch.nn.Dropout(0.1)
        self.layernorm_1 = torch.nn.LayerNorm(d_model)
        self.layernorm_2 = torch.nn.LayerNorm(d_model)
        self.layernorm_3 = torch.nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None, encoder_output: torch.Tensor = None, encoder_mask: torch.Tensor = None):
        '''
        Forward hidden states through an encoder block.

        Args:
            x (torch.Tensor): Hidden states of dimension (batch_size, max_seq_length, d_model).
            attention_mask (torch.Tensor): Mask for auto-regressive property.
            encoder_output (torch.Tensor): Output of the encoder.
            encoder_mask (torch.Tensor): Mask coming from the encoder.

        Returns:
            res (torch.Tensor): Output of the block of dimension (batch_size, max_seq_length, d_model).
        '''
        x_int_1 = self.masked_mha.forward(x, x, x, attention_mask)
        x_int_1 = self.dropout_1(x_int_1)
        x_int_1 = torch.add(x_int_1, x)
        x_int_1 = self.layernorm_1(x_int_1)

        if encoder_output is None:
            x_int_2 = x_int_1
        else:
            x_int_2 = self.mha.forward(x_int_1, encoder_output, encoder_output, attention_mask = encoder_mask)
            x_int_2 = self.dropout_2(x_int_2)
            x_int_2 = torch.add(x_int_2, x_int_1)
            x_int_2 = self.layernorm_2(x_int_2)
        
        res = self.ffn(x_int_2)
        res = self.dropout_3(res)
        res = torch.add(res, x_int_2)
        res = self.layernorm_3(res)

        return res

class Decoder(torch.nn.Module):
    '''
    Decoder part (without the embedding) of a transformer.

    Args:
        n_layers (int): Number of blocks for the decoder.
        d_model (int): Dimension of hidden states.
        d_k (int): Dimension of the linear projection for Q and K.
        d_v (int): Dimension of the linear projection for V.
        h (int): Number of heads.
        d_ff (int): Dimension of inner-layer.

    Attributes:
        blocks (torch.nn.ModuleList): List of encoder blocks.
    '''
    def __init__(self, n_layers: int, d_model: int, d_k: int, d_v: int, h: int, d_ff: int):
        super(Decoder, self).__init__()
        self.n_heads = h
        self.blocks = torch.nn.ModuleList([DecoderBlock(d_model, d_k, d_v, h, d_ff) for _ in range(n_layers)])
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None, encoder_output: torch.Tensor = None, encoder_mask: torch.Tensor = None):
        '''
        Forward inputs embeddings through the encoder.

        Args:
            x (torch.Tensor): Hidden states of dimension (batch_size, max_seq_length, d_model).
            attention_mask (torch.Tensor): Mask for auto-regressive property.
            encoder_output (torch.Tensor): Output of the encoder.
            encoder_mask (torch.Tensor): Mask coming from the encoder.

        Returns:
            res (torch.Tensor): Output of the encoder of dimension (batch_size, max_seq_length, d_model).
        '''
        bs, seq_length, _ = x.shape
        attention_mask = torch.tril(torch.ones(bs, self.n_heads, seq_length, seq_length)).to(device)
        for block in self.blocks:
            x = block.forward(x, attention_mask = attention_mask, encoder_output = encoder_output, encoder_mask = encoder_mask)
        return x