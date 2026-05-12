# AG News Classifier

A text classifier built from scratch in PyTorch trained on the AG News dataset. 
No pretrained models, no shortcuts. Every part of the pipeline is written manually 
— vocabulary, embeddings, architecture, and training loop.

Built as a learning exercise to understand PyTorch fundamentals.

---

## Dataset

AG News — 120,000 training headlines and 7,600 test headlines across 4 categories:

- 0 — World
- 1 — Sports
- 2 — Business
- 3 — Sci/Tech

---

## Architecture

**Embedding layer** — a lookup table where every word in the vocabulary maps to a 
vector of 64 numbers. These vectors start random and get learned during training 
just like any other weight in the network.

**Mean pooling** — all word vectors in a headline are averaged into one single vector 
of 64 numbers. This loses word order but works well for short texts like headlines.

**fc1** — 128 neurons each computing a weighted sum of the 64 input numbers. 
Expands the representation so the network has more capacity to learn complex patterns.

**ReLU** — activation function that zeros out negative neuron outputs. Without it 
fc1 and fc2 would collapse into a single linear operation and the network could 
not learn non-linear patterns.

**fc2** — 4 neurons, one per class. Each outputs a confidence score. 
The highest score is the predicted class.

---

## How to Run
git clone https://github.com/bg20061220/ag-news-classifier
cd ag-news-classifier
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train.py
python evaluate.py
python analysis.py

---

## Training Curves

| Epoch | Loss    | Accuracy |
|-------|---------|----------|
| 1     | 1806.94 | 0.827    |
| 2     | 911.48  | 0.918    |
| 3     | 659.81  | 0.941    |
| 4     | 497.53  | 0.955    |
| 5     | 375.55  | 0.966    |
| 6     | 283.91  | 0.974    |
| 7     | 212.89  | 0.980    |
| 8     | 156.86  | 0.985    |
| 9     | 115.64  | 0.989    |
| 10    | 85.96   | 0.992    |

---

## Results

Test accuracy: **90%**

precision    recall  f1-score   support
   World       0.92      0.89      0.91      1900
  Sports       0.95      0.96      0.95      1900
Business       0.86      0.86      0.86      1900
Sci/Tech       0.85      0.87      0.86      1900
accuracy                           0.90      7600

Sports was the easiest category — sports vocabulary is distinct enough that the 
model rarely confuses it with other categories.

Business and Sci/Tech were the hardest — tech company headlines are often written 
in financial language, making it difficult to separate the two categories.

---

## Failure Analysis

The model achieves 99% training accuracy but 90% test accuracy, indicating slight 
overfitting. The confusion matrix shows three main failure patterns:

**1. World predicted as Sci/Tech (84 cases)**

Geopolitical headlines involving technology get misclassified because the model 
sees tech vocabulary and ignores the political context:

> "US government deploys AI for border surveillance"

The model sees "AI" and predicts Sci/Tech. It has no way to know the subject 
is a government policy decision, not a technology announcement.

**2. Sports vocabulary in non-Sports headlines**

> "Prediction Unit Helps Forecast Wildfires — Lightning will strike... winds will 
> pick up... flames will roar"

The model predicts Sports. Words like "lightning" and "roar" appear frequently 
in sports commentary so the model associates them with Sports regardless of context.

**3. Business vocabulary in Sci/Tech headlines**

> "Google IPO... Wall Street... public bidding"
> "IBM to hire even more new workers"

Google and IBM are tech companies but these headlines are written in financial 
language. Mean pooling averages the tech and business word vectors together and 
the business signal wins.

**Root cause**

All three failure patterns come down to the same limitation — mean pooling treats 
every word equally and throws away word order and context. A headline about the 
government using AI for surveillance has the same average word vector as a headline 
purely about AI research. The model cannot tell them apart.

A transformer based model would handle this better by looking at relationships 
between words rather than averaging them.

---

## What I Learned

- How embedding layers work as learned lookup tables
- Why mean pooling works for short texts but loses contextual nuance
- How cross entropy loss and backpropagation update weights
- How to read a confusion matrix to find failure patterns rather than just reporting accuracy
- That 99% training accuracy means nothing without checking the test set


## Experiments

### Experiment 1 — Learning Rate

Trained for 5 epochs with three different learning rates to observe the effect on convergence.

| Epoch | lr=0.0001 | lr=0.001 | lr=0.01 |
|-------|-----------|----------|---------|
| 1     | 0.578     | 0.827    | 0.892   |
| 2     | 0.763     | 0.918    | 0.949   |
| 3     | 0.813     | 0.941    | 0.967   |
| 4     | 0.843     | 0.955    | 0.978   |
| 5     | 0.862     | 0.966    | 0.984   |

**Insights**

- `lr=0.0001` is too conservative — the model is still climbing at epoch 5 and would need significantly more epochs to converge. The weight updates are so small that the model barely moves each batch.
- `lr=0.001` is solid — steady improvement every epoch, well behaved loss curve.
- `lr=0.01` converges fastest — best accuracy at every epoch. For this problem and architecture, larger update steps actually help rather than causing instability.

The takeaway is that learning rate has a bigger impact on training speed than on final accuracy. Given enough epochs all three would likely converge.

### Experiment 2 — Extra Linear Layer

Added a third linear layer to see if more depth would improve accuracy.

Original architecture: Embedding → Mean Pool → fc1 (64→128) → ReLU → fc2 (128→4)
New architecture: Embedding → Mean Pool → fc1 (64→128) → ReLU → fc2 (128→128) → ReLU → fc3 (128→4)

| Epoch | 2 layers | 3 layers |
|-------|----------|----------|
| 1     | 0.827    | 0.816    |
| 5     | 0.966    | 0.966    |
| 10    | 0.992    | 0.992    |

Test accuracy: **90% on both models**

**Insights**

Adding a third linear layer made no meaningful difference — same training accuracy, 
same test accuracy, same convergence speed. 

The bottleneck in this model is not the depth of the network. It is mean pooling. 
No matter how many linear layers you stack on top, the input to those layers is 
still a single averaged vector that has already lost word order and context. 
Adding more layers cannot recover information that was thrown away during pooling.

To meaningfully improve this classifier the right move is to replace mean pooling 
with a better aggregation method — not to add more linear layers.