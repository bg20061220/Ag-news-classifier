import torch 
from dataset import test_dataset 
from vocabulary import vocab 
from model import model 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("model.pt" , map_location=device))
model = model.to(device)
model.eval()

label_names = ["World" , "Sports" , "Business" , "Sci/Tech"]

wrong_examples = []

with torch.no_grad() : 
    for i in range(500): 
        tensor , true_label = test_dataset[i] 
        tensor = tensor.unsqueeze(0).to(device)
        output = model(tensor) 
        predicted = output.argmax(dim=1).item() 

        if predicted != true_label: 
              wrong_examples.append({
                   "text" : test_dataset.data[i]["text"] , 
                   "true" : label_names[true_label] , 
                   "predicted" : label_names[predicted]
              })


for ex in wrong_examples[:10]: 
        print(f"True: {ex['true']} | Predicted: {ex['predicted']}")
        print(f"Text: {ex['text']}")
        print()
