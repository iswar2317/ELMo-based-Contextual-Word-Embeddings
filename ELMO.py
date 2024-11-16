import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load data
df = pd.read_csv("/kaggle/working/preprocessed_train.csv")
corpus = df["Description"].fillna('')

# Create vocabulary and word2idx
words = set()
for sent in corpus:
    sent = sent.translate(str.maketrans('', '', string.punctuation)).lower()
    words.update([word for word in sent.split() if word.isalpha()])

# Ensure '<unk>' and '<pad>' are in your dictionary
word2idx = {word: i + 2 for i, word in enumerate(sorted(words))}  # Start indexing from 2
word2idx['<unk>'] = 1  # Unknown words
word2idx['<pad>'] = 0  # Padding token
vocab_size=len(word2idx)
# Convert sentences to sequences of indices
def sentence_to_idx(sent):
    return [word2idx.get(word, word2idx['<unk>']) for word in sent if word in word2idx]

processed_inputs = []
processed_outputs = []

for sent in corpus:
    sent = ['<sos>'] + sent.translate(str.maketrans('', '', string.punctuation)).lower().split() + ['<eos>']
    idx_sent = sentence_to_idx(sent)
    processed_inputs.append(torch.tensor(idx_sent[:-1], dtype=torch.long))
    processed_outputs.append(torch.tensor(idx_sent[1:], dtype=torch.long))

# Define the dataset
class TextDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# Collate function to pad sequences
def collate_fn(batch):
    inputs, outputs = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=word2idx['<pad>'])
    outputs_padded = pad_sequence(outputs, batch_first=True, padding_value=word2idx['<pad>'])
    return inputs_padded, outputs_padded

train_dataset = TextDataset(processed_inputs, processed_outputs)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

vocab_size

class ForwardLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_prob):
        super(ForwardLM, self).__init__()
        self.embed_layer = nn.Embedding(vocab_size, embedding_dim)
        self.layer1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.layer2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embed = self.embed_layer(x)
        lstm1, _ = self.layer1(embed)
        lstm1 = self.dropout(lstm1)
        lstm2, _ = self.layer2(lstm1)
        lstm2 = self.dropout(lstm2)
        output = self.fc(lstm2)
        return output  # Returning only output for simplicity in loss calculation

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ForwardLM(vocab_size, 300, 300, 0.5).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])  # Ignore padding in loss computation
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

model.train()
for epoch in range(5):
    running_loss = 0.0
    for inputs, outputs in train_loader:
        inputs, outputs = inputs.to(device), outputs.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output.view(-1, vocab_size), outputs.view(-1))  # Flatten output for CrossEntropyLoss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}: Loss {running_loss / len(train_loader.dataset):.4f}')

# Save the model weights to a file
model_path = '/kaggle/working/forward1.pt'
torch.save(model.state_dict(), model_path)

"""BackwardLM
****
"""

# Assuming you already have processed_inputs and processed_outputs from your previous steps
rev_inputs = [torch.flip(out, [0]) for out in processed_inputs]  # Reverse each input sequence
rev_outputs = [torch.flip(inp, [0]) for inp in processed_outputs]  # Reverse each output sequence

backward_dataset = TextDataset(rev_inputs, rev_outputs)
backward_train_loader = DataLoader(backward_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

class BackwardLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_prob):
        super(BackwardLM, self).__init__()
        self.embed_layer = nn.Embedding(vocab_size, embedding_dim)
        self.layer1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.layer2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed_layer(x)
        x, _ = self.layer1(x)
        x = self.dropout(x)
        x, _ = self.layer2(x)
        x = self.dropout(x)
        return self.fc(x)  # Directly return the output for CrossEntropyLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backward_model = BackwardLM(vocab_size, 300, 300, 0.5).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
optimizer = torch.optim.Adam(backward_model.parameters(), lr=0.0001)

backward_model.train()
for epoch in range(5):
    running_loss = 0.0
    for inputs, outputs in backward_train_loader:
        inputs, outputs = inputs.to(device), outputs.to(device)
        optimizer.zero_grad()
        output = backward_model(inputs)
        loss = criterion(output.view(-1, vocab_size), outputs.view(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}: Loss {running_loss / len(backward_train_loader.dataset):.4f}')

model_path = '/kaggle/working/backward1.pt'
torch.save(backward_model.state_dict(), model_path)