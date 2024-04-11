# Libraries
import torch
import encoder, decoder, embedding

class EncoderOnly(torch.nn.Module):
    '''
    Model with encoder only.

    Args:
        n_layers (int): Number of blocks for the encoder.
        d_model (int): Dimension of hidden states.
        d_k (int): Dimension of the linear projection for Q and K.
        d_v (int): Dimension of the linear projection for V.
        h (int): Number of heads.
        d_ff (int): Dimension of inner-layer.
        vocab_size (int): Dimension of tokenizer's vocabulary.
        pad_token_id (int): Index of padding token.
        max_seq_length (int): Length of input sequences.

    Attributes:
        encoder (torch.nn.Module): Concatenation of Multi-Head Attention + Feed-Forward Networks blocks.
        emb (torch.nn.Module): Embedding and positional encodings for input sequences.
    '''
    def __init__(self, n_layers: int, d_model: int, d_k: int, d_v: int, h: int, d_ff: int, vocab_size: int, pad_token_id: int, max_seq_length: int):
        super(EncoderOnly, self).__init__()
        self.encoder = encoder.Encoder(n_layers, d_model, d_k, d_v, h, d_ff)
        self.emb = embedding.Embedding(vocab_size, d_model, pad_token_id, max_seq_length)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        '''
        Forward input_ids through an encoder-only model.

        Args:
            input_ids (torch.Tensor): Input sequences of id of dimension (batch_size, max_seq_length).
            attention_mask (torch.Tensor): Mask for padding.

        Returns:
            res (torch.Tensor): Output of the encoder-only model of dimension (batch_size, max_seq_length, d_model).
        '''
        x = self.emb.forward(input_ids)
        res = self.encoder.forward(x, attention_mask)
        return res
    
    def get_modules(self):
        '''
        Extract embedding and encoder from a encoder-only architecture.

        Returns:
            self.emb (torch.nn.Module): Embedding and positional encodings layer for input sequences.
            self.encoder (torch.nn.Module): Encoder layer composed of encoder blocks.
        '''
        return self.emb, self.encoder

class DecoderOnly(torch.nn.Module):
    '''
    Model with decoder only.

    Args:
        n_layers (int): Number of blocks for the decoder.
        d_model (int): Dimension of hidden states.
        d_k (int): Dimension of the linear projection for Q and K.
        d_v (int): Dimension of the linear projection for V.
        h (int): Number of heads.
        d_ff (int): Dimension of inner-layer.
        vocab_size (int): Dimension of tokenizer's vocabulary.
        pad_token_id (int): Index of padding token.
        max_seq_length (int): Length of input sequences.

    Attributes:
        decoder (torch.nn.Module): Concatenation of Masked Multi-Head Attention + Multi-Head Attention + Feed-Forward Networks blocks.
        emb (torch.nn.Module): Embedding and positional encodings for input sequences.
        classifier (torch.nn.Module): Linear layer to ids space.
    '''
    def __init__(self, n_layers: int, d_model: int, d_k: int, d_v: int, h: int, d_ff: int, vocab_size: int, pad_token_id: int, max_seq_length: int):
        super(DecoderOnly, self).__init__()
        self.decoder = decoder.Decoder(n_layers, d_model, d_k, d_v, h, d_ff)
        self.emb = embedding.Embedding(vocab_size, d_model, pad_token_id, max_seq_length)
        self.classifier = torch.nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        '''
        Forward input_ids through a decoder-only model.

        Args:
            input_ids (torch.Tensor): Input sequences of id of dimension (batch_size, max_seq_length).
            attention_mask (torch.Tensor): Mask for auto-regressive property.

        Returns:
            res (torch.Tensor): Output of the decoder-only model of dimension (batch_size, max_seq_length, vocab_size).
        '''
        x = self.emb.forward(input_ids)
        res = self.decoder.forward(x, attention_mask)
        res = self.classifier(res)
        return res
    
    def get_modules(self):
        '''
        Extract embedding, decoder and classifier from a decoder-only architecture.

        Returns:
            self.emb (torch.nn.Module): Embedding and positional encodings layer for input sequences.
            self.decoder (torch.nn.Module): Decoder layer composed of decoder blocks.
            self.classifier (torch.nn.Linear): Linear layer to ids space.
        '''
        return self.emb, self.decoder, self.classifier


class EncoderDecoder(torch.nn.Module):
    '''
    Model with an encoder and a decoder (i.e a transformer as in "Attention is all you need").

    Args:
        enc (torch.nn.Module): An encoder.
        dec (torch.nn.Module): A decoder.
        enc_emb (torch.nn.Module): Encoder's embedding.
        dec_emb (torch.nn.Module): Decoder's embedding.
        classifier (torch.nn.Linear): Linear layer to ids space.
    '''
    def __init__(self, enc: torch.nn.Module, enc_emb: torch.nn.Module, dec: torch.nn.Module, dec_emb: torch.nn.Module, classifier: torch.nn.Linear):
        super(EncoderDecoder, self).__init__()
        self.encoder = enc
        self.decoder = dec
        self.encoder_emb = enc_emb
        self.decoder_emb = dec_emb
        self.classifier = classifier
    
    def forward(self, encoder_ids: torch.Tensor, decoder_ids: torch.Tensor, encoder_mask: torch.Tensor = None, decoder_mask: torch.Tensor = None):
        '''
        Forward inputs through the model.

        Args:
            encoder_ids (torch.Tensor): Input sequences of id of dimension (batch_size, max_seq_length) for the encoder.
            encoder_mask (torch.Tensor): Mask for padding.
            decoder_ids (torch.Tensor): Input sequences of id of dimension (batch_size, max_seq_length) for the decoder.
            decoder_mask (torch.Tensor): Mask for auto-regressive property.

        Returns:
            res (torch.Tensor): Output of the encoder-decoder model of dimension (batch_size, max_seq_length, vocab_size).
        '''
        x_enc = self.encoder_emb.forward(encoder_ids)
        x_dec = self.decoder_emb.forward(decoder_ids)
        encoder_output = self.encoder.forward(x_enc, attention_mask = encoder_mask)
        decoder_output = self.decoder.forward(x_dec, attention_mask = decoder_mask, encoder_output = encoder_output, encoder_mask = encoder_mask)
        res = self.classifier(decoder_output)
        return res