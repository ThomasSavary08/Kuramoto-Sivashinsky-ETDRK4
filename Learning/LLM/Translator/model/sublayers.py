# Libraries
import math
import torch

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

class ScaledDotProductAttention(torch.nn.Module):
    '''
    Scaled Dot-Product Attention used in Multi-Head Attention.

    Args:
        d_k (int): Dimension of the linear projection for Q and K.

    Attributes:
        softmax (torch.nn.Softmax): Softmax function for Scaled Dot-Product Attention.
        scale_factor (torch.Tensor): Scaling factor to avoid small gradients.
    '''
    def __init__(self, d_k: int):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.softmax = torch.nn.Softmax(dim = -1)
        self.scale_factor = torch.tensor([1./math.sqrt(self.d_k)]).to(device)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor = None):
        '''
        Compute Scaled Dot-Product Attention for q, k and v as inputs.

        Args:
            q (torch.Tensor): Queries.
            k (torch.Tensor): Keys.
            v (torch.Tensor): Values.
            attention_mask (torch.Tensor): Mask for padding and/or auto-regressive property.

        Returns:
            res (torch.Tensor): Output of the Scaled Dot-Product Attention.
        '''
        score = torch.matmul(q, torch.transpose(k, -1, -2))
        score = torch.mul(score, self.scale_factor)
        if attention_mask is not None:
            if (attention_mask.dim() == 2):
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            score = score.masked_fill(attention_mask == 0, -math.inf)
        res = torch.matmul(self.softmax(score), v)
        return res

class MHA(torch.nn.Module):
    '''
    Multi-Head Attention sublayer.

    Args:
        d_model (int): Dimension of hidden states.
        d_k (int): Dimension of the linear projection for Q and K.
        d_v (int): Dimension of the linear projection for V.
        h (int): Number of heads.

    Attributes:
        proj_(Q/K/V/O) (torch.nn.Linear): Linear projectors.
        scaled_dpa (ScaledDotProductAttention): Scaled Dot-Product Attention.
    '''
    def __init__(self, d_model: int, d_k: int, d_v: int, h: int):
        super(MHA, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.proj_Q = torch.nn.Linear(d_model, h*d_k, bias = False)
        self.proj_K = torch.nn.Linear(d_model, h*d_k, bias = False)
        self.proj_V = torch.nn.Linear(d_model, h*d_v, bias = False)
        self.proj_O = torch.nn.Linear(h*d_v, d_model, bias = False)
        self.scaled_dpa = ScaledDotProductAttention(d_k)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor = None):
        '''
        Forward q, k and v through the Multi-Head Attention layer.

        Args:
            q (torch.Tensor): Queries.
            k (torch.Tensor): Keys.
            v (torch.Tensor): Values.
            attention_mask (torch.Tensor): Mask for padding and/or auto-regressive property.

        Returns:
            res (torch.Tensor): Output of the Multi-Head Attention.
        '''
        # Project Q, K and V
        q, k, v = self.proj_Q(q), self.proj_K(k), self.proj_V(v)


        # Reshape tensors
        bs, length, _ = q.shape
        q = torch.transpose(q.view(bs, length, self.h, self.d_k), 1, 2)
        k = torch.transpose(k.view(bs, length, self.h, self.d_k), 1, 2)
        v = torch.transpose(v.view(bs, length, self.h, self.d_v), 1, 2)

        # Scaled Dot-Product Attention
        res = self.scaled_dpa.forward(q, k, v, attention_mask)

        # Concat
        res = torch.transpose(res, 1, 2).contiguous().view(bs, length, self.h*self.d_v)

        # Project result
        res = self.proj_O(res)
        return res
    
class FFN(torch.nn.Module):
    '''
    Position-wise Feed-Forward Networks.

    Args:
        d_model (int): Dimension of hidden states.
        d_ff (int): Dimension of inner-layer.

    Attributes:
        lin_(1/2) (torch.nn.Linear): Linear layers.
        relu (torch.nn.ReLU): Activation function.
    '''
    def __init__(self, d_model: int, d_ff: int):
        super(FFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.lin_1 = torch.nn.Linear(d_model, d_ff)
        self.lin_2 = torch.nn.Linear(d_ff, d_model)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x: torch.Tensor):
        '''
        Forward an input through the Position-wise Feed-Forward Network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            res (torch.Tensor): Output of the network equals to max(0, xW1 + b1)W2 + b2
        '''
        res = self.lin_1(x)
        res = self.relu(res)
        res = self.lin_2(res)
        return res
