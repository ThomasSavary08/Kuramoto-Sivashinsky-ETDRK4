# Set path
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './', 'model')))

# Libraries
import math
import tqdm
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import CamembertTokenizerFast, T5TokenizerFast
from model.transformer import EncoderOnly, DecoderOnly, EncoderDecoder

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model parameters
n_layers = 6
d_model = 512 
d_k = 64 
d_v = 64 
h = 8
d_ff = 2048
max_seq_length = 256

# Load dataset and tokenizers
translation_ds = load_dataset('wmt14', 'fr-en', split = 'train', trust_remote_code = True)
french_tokenizer = CamembertTokenizerFast.from_pretrained('./pretraining/encoder/french_tokenizer')
english_tokenizer = T5TokenizerFast.from_pretrained('./pretraining/decoder/english_tokenizer')

# Embedding parameters
vocab_size_fr, vocab_size_en = french_tokenizer.vocab_size, english_tokenizer.vocab_size
pad_token_id_fr, pad_token_id_en = french_tokenizer.pad_token_id, english_tokenizer.pad_token_id

# Load encoder
encoder = EncoderOnly(n_layers, d_model, d_k, d_v, h, d_ff, vocab_size_fr, pad_token_id_fr, max_seq_length)
encoder.load_state_dict(torch.load('./pretraining/encoder/pretrained_encoder.pth'))
enc_embedding, enc = encoder.get_modules()

# Load decoder
decoder = DecoderOnly(n_layers, d_model, d_k, d_v, h, d_ff, vocab_size_en, pad_token_id_en, max_seq_length)
decoder.load_state_dict(torch.load('./pretraining/decoder/pretrained_decoder.pth'))
dec_embedding, dec, classifier = decoder.get_modules()

# Instanciate transformer for translation from french to english
model = EncoderDecoder(enc, enc_embedding, dec, dec_embedding, classifier).to(device)

# Training parameters and objects
batch_size = 64
label_smoothing = 0.05
ignore_index = pad_token_id_en
criterion = torch.nn.CrossEntropyLoss(ignore_index = ignore_index, reduction = 'mean', label_smoothing = label_smoothing)

# Processing function
def process_inputs(batch):
    to_translate = batch['translation']
    french_texts = [to_translate[i]['fr'] for i in range(len(to_translate))]
    english_texts = [to_translate[i]['en'] for i in range(len(to_translate))]
    french_tokens = french_tokenizer(french_texts, padding = 'max_length', max_length = max_seq_length, truncation = True, 
                       return_attention_mask = True, return_tensors = 'pt')
    english_tokens = english_tokenizer(english_texts, padding = 'max_length', max_length = max_seq_length, truncation = True, 
                       return_attention_mask = True, return_tensors = 'pt')
    shifted_input_ids = torch.full(english_tokens['input_ids'].shape[:-1] + (1,), pad_token_id_en)
    shifted_input_ids = torch.cat([shifted_input_ids, english_tokens['input_ids'][..., :-1]], dim = -1)
    return {'enc_input_ids': french_tokens['input_ids'], 'enc_attention_mask': french_tokens['attention_mask'], 
            'dec_input_ids': shifted_input_ids, 'dec_attention_mask': english_tokens['attention_mask'],
            'labels': english_tokens['input_ids']}

# Set dataset and instanciate dataloader
translation_ds.set_transform(process_inputs)
dataloader = torch.utils.data.DataLoader(translation_ds, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = 4)

# Optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr = 1./math.sqrt(d_model))
def lambda_lr(current_step: int):
    step_num = current_step + 1
    warmup_steps = 4000
    a, b, c = 1./math.sqrt(d_model), 1./math.sqrt(step_num), 1./(warmup_steps*math.sqrt(warmup_steps))
    return float(a*min(b,step_num*c))
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda = lambda_lr, last_epoch = -1)

# Loop on dataloader batches to optimize network parameters
loss_list = []
tqdm_train = tqdm.tqdm(dataloader, total = int(len(dataloader)))
for _, inputs in enumerate(tqdm_train):
        
    # Training mode
    model.train()

    # Clean gradients
    optimizer.zero_grad()

    # Compute loss
    enc_input_ids, enc_attention_mask = inputs['enc_input_ids'], inputs['enc_attention_mask']
    dec_input_ids, dec_attention_mask, labels = inputs['dec_input_ids'], inputs['dec_attention_mask'], inputs['labels']
    enc_input_ids, enc_attention_mask = enc_input_ids.to(device), enc_attention_mask.to(device) 
    dec_input_ids, dec_attention_mask, labels = dec_input_ids.to(device), dec_attention_mask.to(device), labels.to(device)
    with torch.set_grad_enabled(True):
        logits = model.forward(enc_input_ids, dec_input_ids, enc_attention_mask, dec_attention_mask)
        loss = criterion(logits.view(-1, vocab_size_en), labels.view(-1))
        loss.backward()

    # Update parameters
    optimizer.step()

    # Update loss list
    loss_list.append(loss.item())
    tqdm_train.set_postfix(loss = loss.item())
    
	# Update learning rate
    lr_scheduler.step()

# Save the pretrained encoder
torch.save(model.state_dict(), './trained_transformer.pth')

# Plot and save loss evolution
plt.figure()
plt.plot(loss_list)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Evolution of loss function during transformer fine-tuning")
plt.savefig('./finetuning_loss.png')
plt.show()