from datasets import load_dataset 

dataset = load_dataset("ag_news") 
print(dataset) 

label_names =[ 0 , 1  , 2 , 3] 

for target_label in range(4): 
    for example in dataset["train"] : 
        if example["label"] == target_label : 
            print(f"Label {target_label} ({label_names[target_label]}):")
            print(example["text"]) 
            print()
            break 
