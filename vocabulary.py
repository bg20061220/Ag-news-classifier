from datasets import load_dataset 
from collections import Counter 

dataset = load_dataset("ag_news")

def tokenize(text): 
    return text.lower().split() 

counter = Counter() 
for example in dataset["train"] : 
    counter.update(tokenize(example["text"]))

vocab = {"<pad>" : 0 , "<unk>" : 1 } 
for word , count in counter.items() : 
    if count >= 2 : 
        vocab[word] = len(vocab)

print(f"Vocablury size : {len(vocab)}") 

