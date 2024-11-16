import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from tqdm import tqdm
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

vocab_size

df = pd.read_csv('/kaggle/working/preprocessed_train.csv')
class_index = df['Class Index'] - 1  # convert classes from 1-4 to 0-3
descriptions = df['Description'].fillna('')

# Assume 'word2idx' is available from earlier
inputs = [torch.tensor([word2idx.get(word, word2idx['<unk>']) for word in sent.split()], dtype=torch.long) for sent in descriptions]
labels = torch.tensor(class_index.values, dtype=torch.long)

# Custom dataset
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    texts, labels = zip(*batch)
    # Pad the sequences to have the same length
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=word2idx['<pad>'])
    labels = torch.stack(labels)  # Ensure labels are also properly formatted as a tensor
    return texts_padded, labels

# Assuming that your dataset and DataLoader are set up as follows:
class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# DataLoader
dataset = CustomDataset(inputs, labels)
loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from tqdm import tqdm

class ForwardLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_prob):
        super(ForwardLM, self).__init__()
        self.embed_layer = nn.Embedding(vocab_size, embedding_dim)
        self.layer1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.layer2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed_layer(x)
        x, (h1, c1) = self.layer1(x)
        x = self.dropout(x)
        x, (h2, c2) = self.layer2(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x, h2  # Return sequence output and last hidden state


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
        x, (h1, c1) = self.layer1(x)
        x = self.dropout(x)
        x, (h2, c2) = self.layer2(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x, h2  # Return sequence output and last hidden state

class DownstreamModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, forwardLM, backwardLM, num_lam):
        super(DownstreamModel, self).__init__()
        self.forwardLM = forwardLM
        self.backwardLM = backwardLM
        self.layer = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.lambdas = nn.Parameter(torch.rand(num_lam))

    def forward(self, xf, xb):
        fout, fhidden = self.forwardLM(xf)  # Assume these return the output and the last hidden state
        bout, bhidden = self.backwardLM(xb)

        # Ensure hidden states are properly extracted if not, change indices accordingly
        fhidden = fhidden[-1]  # Assuming fhidden is the last layer's hidden state
        bhidden = bhidden[-1]  # Assuming bhidden is the last layer's hidden state
        combined = torch.cat((fhidden, bhidden), dim=1)
        out, _ = self.layer(combined)  # Ensure combined has shape [batch_size, seq_len, features]
        out = self.output(out.squeeze(1))  # Adjust this depending on whether you have a seq_len dimension

        return self.softmax(out)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize language models
forward_lm = ForwardLM(vocab_size, 300, 300, 0.5).to(device)
backward_lm = BackwardLM(vocab_size, 300, 300, 0.5).to(device)

# Load pretrained weights
forward_lm.load_state_dict(torch.load('/kaggle/working/forward1.pt'))
backward_lm.load_state_dict(torch.load('/kaggle/working/backward1.pt'))

# Freeze the language models to prevent further training
for param in forward_lm.parameters():
    param.requires_grad = True
for param in backward_lm.parameters():
    param.requires_grad = True

# Setup the downstream model
downstream_model = DownstreamModel(600, 300, 4, 1, forward_lm, backward_lm, 3).to(device)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(downstream_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
downstream_model.train()
num_epochs = 5

for epoch in range(num_epochs):
    total_loss = 0
    for texts, labels in tqdm(loader):
        texts, labels = texts.to(device), labels.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        logits = downstream_model(texts, torch.flip(texts, dims=[1]))  # flipping for backward model input

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Logging the losses
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(loader)}')

print(downstream_model.lambdas)

# Save the trained model
model_path = '/kaggle/working/classifier.pt'
torch.save(downstream_model.state_dict(), model_path)

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the test data
df_test = pd.read_csv('/kaggle/working/lower_test.csv')
class_index_test = df_test['Class Index'] - 1  # Adjust class index as necessary
descriptions_test = df_test['Description'].fillna('')

# Prepare inputs and labels for the test data
test_inputs = [torch.tensor([word2idx.get(word, word2idx['<unk>']) for word in sent.split()], dtype=torch.long) for sent in descriptions_test]
test_labels = torch.tensor(class_index_test.values, dtype=torch.long)

# DataLoader for test data
def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=word2idx['<pad>'])
    labels = torch.tensor(labels, dtype=torch.long)  # Ensure labels are properly formatted
    return texts_padded, labels

test_dataset = CustomDataset(test_inputs, test_labels)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the test data
df_test = pd.read_csv('/kaggle/working/preprocessed_train.csv')
class_index_test = df_test['Class Index'] - 1  # Adjust class index as necessary
descriptions_test = df_test['Description'].fillna('')

# Prepare inputs and labels for the test data
test_inputs = [torch.tensor([word2idx.get(word, word2idx['<unk>']) for word in sent.split()], dtype=torch.long) for sent in descriptions_test]
test_labels = torch.tensor(class_index_test.values, dtype=torch.long)

# DataLoader for test data
def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=word2idx['<pad>'])
    labels = torch.tensor(labels, dtype=torch.long)  # Ensure labels are properly formatted
    return texts_padded, labels

test_dataset = CustomDataset(test_inputs, test_labels)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'downstream_model' and 'test_loader' are already defined and properly set up
downstream_model.to(device)
downstream_model.eval()

# Evaluate the model
predicted_labels = []
true_labels = []

with torch.no_grad():
    for texts, labels in tqdm(test_loader):
        texts, labels = texts.to(device), labels.to(device)
        outputs = downstream_model(texts, torch.flip(texts, dims=[1]))  # Assuming model takes normal and flipped inputs
        _, predicted = torch.max(outputs.data, 1)
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Calculate accuracy and other metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')
conf_matrix = confusion_matrix(true_labels, predicted_labels)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=['Class 1', 'Class 2', 'Class 3', 'Class 4'], yticklabels=['Class 1', 'Class 2', 'Class 3', 'Class 4'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()