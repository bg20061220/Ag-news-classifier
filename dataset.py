import torch 
from torch.utils.data import Dataset , DataLoader 
from torch.nn.utils.rnn import pad_sequence 
from vocabulary import vocab , tokenize 
from datasets import load_dataset 

dataset = load_dataset("ag_news")

class NewsDataset(Dataset) : 
    def __init__(self , data , vocab) : 
        self.data = data 
        self.vocab = vocab 

    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self , idx):
        example = self.data[idx] 
        tokens = tokenize(example["text"])
        indices = [self.vocab.get(token , 1) for token in tokens] 
        return torch.tensor(indices) , example["label"]
    

train_dataset = NewsDataset(dataset["train"] , vocab)
test_dataset = NewsDataset(dataset["test"] , vocab)  

def collate_fn(batch): 
    texts , labels = zip(*batch) 
    texts = pad_sequence(texts , batch_first=True , padding_value=0) 
    labels = torch.tensor(labels) 
    return texts , labels 

train_loader = DataLoader(train_dataset , batch_size=32 , shuffle=True , collate_fn=collate_fn)
test_loader = DataLoader(test_dataset , batch_size=32 , shuffle=False , collate_fn=collate_fn)

texts , labels = next(iter(train_loader)) 
print(f"Batch text shape: {texts.shape}")
print(f"Batch labels shape: {labels.shape}")

