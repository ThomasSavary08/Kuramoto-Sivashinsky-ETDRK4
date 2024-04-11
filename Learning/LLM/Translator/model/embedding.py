# Libraries
import torch

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

def getPositionalEncoding(d_model: int, seq_length: int):
    '''
    Compute "positional encodings" matrix of dimension (seq_length, d_model).

    Args:
        d_model (int): Dimension of hidden states.
        seq_length (int): Length of input sequences.

    Returns:
        pe (torch.Tensor): Tensor containing "positional encodings".
    '''
    pe = torch.zeros((seq_length, d_model), requires_grad = False)
    pos = torch.arange(0, seq_length, 1, dtype = torch.float).unsqueeze(1)
    dim = torch.arange(0, d_model, 2, dtype = torch.float)
    pe[:, 0::2] = torch.sin(pos/(10000**(dim/d_model)))
    pe[:, 1::2] = torch.cos(pos/(10000**(dim/d_model)))
    return pe

class Embedding(torch.nn.Module):
    '''
    Input/Output tokens embedding.

    Args:
        vocab_size (int): Dimension of tokenizer's vocabulary.
        d_model (int): Dimension of hidden states.
        pad_token_id (int): Index of padding token.
        max_seq_length (int): Length of input sequences.

    Attributes:
        emb (torch.nn.Embedding): Softmax function for Scaled Dot-Product Attention.
        pe (torch.Tensor): Positional encodings.
        dropout (torch.nn.Dropout): Dropout module.
    '''
    def __init__(self, vocab_size: int, d_model: int, pad_token_id: int, max_seq_length: int):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.emb = torch.nn.Embedding(vocab_size, d_model, padding_idx = pad_token_id)
        self.pe = getPositionalEncoding(d_model, max_seq_length).to(device)
        self.dropout = torch.nn.Dropout(p = 0.1)

    def forward(self, input_ids: torch.Tensor):
        '''
        Forward input sequences through embedding layer.

        Args:
            input_ids (torch.Tensor): Input sequences of id of dimension (batch_size, max_seq_length).

        Returns:
            res (torch.Tensor): Output of the embedding layer of dimension (batch_size, max_seq_length, d_model).
        '''
        bs, seq_length = input_ids.shape
        assert (seq_length == self.max_seq_length)
        res = self.dropout(torch.add(self.emb(input_ids), self.pe.unsqueeze(0).repeat(bs, 1, 1)))
        return res