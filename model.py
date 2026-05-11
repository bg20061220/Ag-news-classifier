from data.vocabulary import vocab
import torch 
import torch.nn as nn 

class NewsClassifier(nn.Module): 
    def __init__(self , vocab_size , embedding_dim , hidden_dim , num_classes): 
        super().__init__() 
        self.embedding = nn.Embedding(vocab_size , embedding_dim , padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim , hidden_dim) 
        self.relu = nn.Relu()
        self.fc2 = nn.Linear(hidden_dim , num_classes) 

    def forward(self , x): 
        embedded = self.embedding(x) 
        pooled = embedded.mean(dim = 1)
        out = self.fc1(pooled) 
        out = self.relu(out)
        out = self.fc2(out) 
        return out 

model = NewsClassifier(
    vocab_size=len(vocab),
    embedding_dim=64, 
    hidden_dim=128,
    num_classes=4
) 

print(model)