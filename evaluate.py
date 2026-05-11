import torch 
import torch.nn as nn 
from sklearn.metrics import confusion_matrix , classification_report 
import matplotlib.pyplot as plt 
import seaborn as sns 
from dataset import test_loader 
from model import model 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load("model.pt"))
model = model.to(device) 
model.eval() 

all_predictions = []
all_labels = [] 

with torch.no_grad():  
    for texts , labels in test_loader : 
        texts = texts.to(device)
        labels = labels.to(device)

        predictions = model(texts) 
        predicted_classes = predictions.argmax(dim=1)

        all_predictions.extend(predicted_classes.cpu().numpy()) 
        all_labels.extend(labels.cpu().numpy()) 

label_names = ["World" , "Sports" , "Business" , "Sci/Tech"]

print(classification_report(all_labels , all_predictions , target_names=label_names))

cm = confusion_matrix(all_labels , all_predictions)
plt.figure(figsize=(8 , 6))
sns.heatmap(cm  , annot = True , fmt = "d" , xticklabels=label_names , yticklabels=label_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show() 

