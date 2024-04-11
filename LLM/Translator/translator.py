# Set path
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './', 'model')))

# Libraries
import torch
from transformers import CamembertTokenizerFast, T5TokenizerFast
from model.transformer import EncoderOnly, DecoderOnly, EncoderDecoder

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Parameters
n_layers = 6
d_model = 512 
d_k = 64 
d_v = 64 
h = 8
d_ff = 2048
max_seq_length = 256

# Create a translator
class translator():
    '''
    French to english translator.

    Attributes:
        french_tokenizer (transformers.CamembertTokenizerFast): French tokenizer.
        english_tokenizer (transformers.T5TokenizerFast): English tokenizer.
        vocab_size_(fr/en) (int): Size of tokenizers vocabulary.
        pad_token_id_(fr/en) (int): Padding index of tokenizers.
        eos_token_id (int): Index of the end-of-the-sentence token (</s>).
        model (torch.nn.Module): Fine-tuned transformer on wmt14 for translation from french to english.
    '''
    def __init__(self):
        self.french_tokenizer = CamembertTokenizerFast.from_pretrained('./pretraining/encoder/french_tokenizer')
        self.english_tokenizer = T5TokenizerFast.from_pretrained('./pretraining/decoder/english_tokenizer')
        self.vocab_size_fr, self.vocab_size_en = self.french_tokenizer.vocab_size, self.english_tokenizer.vocab_size
        self.pad_token_id_fr, self.pad_token_id_en = self.french_tokenizer.pad_token_id, self.english_tokenizer.pad_token_id
        self.eos_token_id = self.english_tokenizer.eos_token_id
        encoder = EncoderOnly(n_layers, d_model, d_k, d_v, h, d_ff, self.vocab_size_fr, self.pad_token_id_fr, max_seq_length)
        enc_embedding, enc = encoder.get_modules()
        decoder = DecoderOnly(n_layers, d_model, d_k, d_v, h, d_ff, self.vocab_size_en, self.pad_token_id_en, max_seq_length)
        dec_embedding, dec, classifier = decoder.get_modules()
        self.model = EncoderDecoder(enc, enc_embedding, dec, dec_embedding, classifier).to(device)
        self.model.load_state_dict(torch.load('./trained_transformer.pth'))
        self.model.eval()
    
    def greedy_search(self, french_input_ids: torch.Tensor, french_attention_mask: torch.Tensor):
        '''
        Generate a translation with greedy search algorithm.

        Args:
            french_input_ids (torch.Tensor): French input_ids corresponding to the tokenization of the input sentence.
            french_attention_mask (torch.Tensor): Attention mask associated with french_input_ids.

        Returns:
            english_output_ids (List[int]): English output_ids corresponding to the translation of the output sentence.
        '''
        with torch.no_grad():
            english_input_ids = torch.mul(self.pad_token_id_en, torch.ones(1, max_seq_length, requires_grad = False)).to(device, dtype = torch.long)
            english_output_ids = []
            eos, ind = False, 0
            while (not eos) and (ind < (max_seq_length - 1)):
                logits = self.model.forward(french_input_ids, english_input_ids, french_attention_mask, decoder_mask = None)
                probas = torch.nn.functional.softmax(logits, dim = -1)
                _, top_id = torch.topk(probas, k = 1, dim = -1)
                top_id = top_id.squeeze(0).squeeze(-1)[ind].item()
                if top_id == self.eos_token_id:
                    eos = True
                else:
                    english_output_ids.append(top_id)
                    english_input_ids[0, ind + 1] = top_id
                    ind += 1
            return english_output_ids
    
    def translate(self, french_input_sentence: str, method: str = 'greedy'):
        '''
        Translate an input french sentence to english.

        Args:
            french_input_sentence (str): The sentece (in french) to translate.
            method (str): Generation algorithm to use for the translation ('greedy' or 'beam').

        Returns:
            english_output_sentence (str): English translation of the input sentence.
        '''
        assert (method == 'greedy') or (method == 'beam')
        french_tokens = self.french_tokenizer([french_input_sentence], padding = 'max_length', max_length = max_seq_length, truncation = True, 
                       return_attention_mask = True, return_tensors = 'pt')
        french_input_ids = french_tokens['input_ids'].to(device)
        french_attention_mask = french_tokens['attention_mask'].to(device)
        if (method == 'greedy'):
            english_output_ids = self.greedy_search(french_input_ids , french_attention_mask)
        elif (method == 'beam'):
            english_output_ids = self.beam_search(french_input_ids, french_attention_mask)
        english_output_sentence = self.english_tokenizer.decode(english_output_ids)
        return english_output_sentence