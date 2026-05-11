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

# model.py 
the actual model shit we use the nn.Module to inherit the model class ,then we do super__init__() to get the inner pytorch machinery running after that there is self.embedding what that does is creates a lookup table , with vocab len number of rows and embedding number of columns 64 
and they are filled with random values but whenever the model gets a tensor with the indices it gts the 64 vector from the table and its also a weight keeps getting adjusted.

After the embedding layer , when you input a headline there is one vector per word and if it has 5 words it has 5 vectors of 64 numbers each  then we average all 5 vectors into one single vector of 64 numbers and then that vector goes into fc1 

 the 64 number vector is passed to 128 neurons with its own set of weights and a final output is passed after a the mx+bias formula for each neuron.

ReLu is the activation function.  it decides which neurons fires and which doesn't based on the input and the computed output. If the neuron output is less than 0 its turned to 0 or saying it doesn't fire if its positive that value is passed on. WIthout it , the two linear operations would just become one linear opeartion making it harder to capture complex patterns. 

and then the self.fc2 does a similar thing where the 128 vector is the input and it passed on to 4 neurons of 128 weights with each neuron per class and they represent how confident they are about the class it belongs to.

This model has 3 layers embeddings -> fc1 ->fc2 

# train.py
Adam is the algorithm that updates the weights after each batch. nn.Module keeps tracks of all weights on all three layers. learning rate is basically how much of an update it can do in one execution.

CrossEntropyLoss measures how wrong the prediction is. It converts the 4 label scores into proababilities adding to 1 and the most high one with the true label , by log(prob) if wrong and -log(prob) if positive and this happens for all 32 text and the returned value is the average of that and that is what loss.backward works with. 

oh so the loss.backward() computes all the gradients and the optimizer.step() actually uses those gradients to update their weights. 