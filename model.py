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