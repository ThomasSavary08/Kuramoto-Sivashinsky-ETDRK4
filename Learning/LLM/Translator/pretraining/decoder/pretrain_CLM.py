# Set path
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'model')))

# Libraries
import math
import tqdm
import torch
import transformer
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import T5TokenizerFast

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

# Load dataset and tokenizer
english_ds = load_dataset('wikimedia/wikipedia', '20231101.en', split = 'train[:50%]')
tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-base', model_max_length = max_seq_length)

# Embedding parameters
vocab_size = tokenizer.vocab_size
pad_token_id = tokenizer.pad_token_id

# Instanciate model for Masked Language Modeling
decoder = transformer.DecoderOnly(n_layers, d_model, d_k, d_v, h, d_ff, vocab_size, pad_token_id, max_seq_length).to(device)

# Training parameters and objects
batch_size = 64
ignore_index = pad_token_id
label_smoothing = 0.05
criterion = torch.nn.CrossEntropyLoss(ignore_index = ignore_index, reduction = 'mean', label_smoothing = label_smoothing)

# Processing function (taken from Hugging Face source code)
def process_inputs(batch):
    tokens = tokenizer(batch['text'], padding = 'max_length', max_length = max_seq_length, truncation = True, 
                       return_attention_mask = True, return_tensors = 'pt')
    shifted_input_ids = torch.full(tokens['input_ids'].shape[:-1] + (1,), pad_token_id)
    shifted_input_ids = torch.cat([shifted_input_ids, tokens['input_ids'][..., :-1]], dim = -1)
    return {'input_ids': shifted_input_ids, 'attention_mask': tokens['attention_mask'], 'labels': tokens['input_ids']}

# Set dataset and instanciate dataloader
english_ds.set_transform(process_inputs)
dataloader = torch.utils.data.DataLoader(english_ds, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = 4)

# Optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(decoder.parameters(), lr = 1./math.sqrt(d_model))
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
    decoder.train()

    # Clean gradients
    optimizer.zero_grad()

    # Compute loss
    input_ids, attention_mask, labels = inputs['input_ids'], inputs['attention_mask'], inputs['labels']
    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
    with torch.set_grad_enabled(True):
        logits = decoder.forward(input_ids = input_ids, attention_mask = attention_mask)
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        loss.backward()

    # Update parameters
    optimizer.step()

    # Update loss list
    loss_list.append(loss.item())
    tqdm_train.set_postfix(loss = loss.item())
    
	# Update learning rate
    lr_scheduler.step()

# Save the pretrained decoder
torch.save(decoder.state_dict(), './pretrained_decoder.pth')

# Plot and save loss evolution
plt.figure()
plt.plot(loss_list)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Evolution of loss function during decoder pretraining")
plt.savefig('./decoder_loss.png')
plt.show()