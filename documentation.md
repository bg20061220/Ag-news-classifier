# explore.py
Exploring the data table 
Data got from Datasets library from Hugging face , really simple layout
text and label.

Labels are 0 1 2 3 
These prolly correspond to 0 to World, 1 to Sport, 2 to Business, and 3 to Science and Tech. 


# Vocabulary.py
Wrote vocablury.py by building a dictionary of all the words appearing more than once and giving each word a unique index , since neural networks can only work with numbers. 
Also added <pad> to make every tensor the same length when we train the neural network and <unk> so that if while testing , it sees something it hasn't seen before it maps that to 1 instead of crashing. 


# Dataset.py
Writing this to prepare the data to feed the Data Loader in pytorch , 
its necessary to wrap the data in a class that answers two questiosn , number of items and get_item number 

# Dataloader.py 
extension work of dataset.py picks 32 random samples from the data  makes sure they are the same length and feeds it to the model and keeps doing it untill we have completed the training. The examples are picked randomly during training and sequentially during testing and they don't need to be the same length across batches. 