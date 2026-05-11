import torch 
import torch.nn as nn 
from dataset import train_loader 
from model import model 

optimzer = torch.optim.Adam(model.parameters() , lr = 0.001) 
criterion = nn.CrossEntropyLoss() 

def train(num_epochs = 10): 
    for epoch in range(num_epochs):
        model.train() 
        total_loss = 0 
        correct = 0 
        total = 0 

        for texts , labels in train_loader: 
            predictions = model(texts) 
            loss = criterion(predictions , labels) 

            optimzer.zero_grad() # this is here to make sure that the gradients are reset before backpropagation
            loss.backward()
            optimzer.step() 

            total_loss += loss.item() 
            predicted_classes = predictions.argmax(dim=1)
            correct += (predicted_classes == labels).sum().item()
            total += labels.size(0)

    accuracy = correct/total
    print(f"Epoch {epoch+1}: Loss={total_loss:.3f} Accuracy={accuracy:.3f}")

train() 
