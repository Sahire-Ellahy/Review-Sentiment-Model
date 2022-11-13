import math
import torch
import torch.nn as nn
import numpy as np

import os
import pandas as pd
from torchvision.io import read_image

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import attentionlayer

from torch.utils.tensorboard import SummaryWriter #Using TensorBoard in PyTorch: https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html


# sentiment labeled sentences
# https://www.kaggle.com/datasets/marklvl/sentiment-labelled-sentences-data-set

# make dataloader and load in above dataset, use https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class SentimentSentencesDataset(Dataset):
    def __init__(self, amazon_cells_labelled, imdb_labelled, yelp_labelled, skiprows=None, nrows=None):
        self.amazon_cells_labelled = pd.read_csv(amazon_cells_labelled, skiprows=skiprows, nrows=nrows)
        self.imdb_labelled = pd.read_csv(imdb_labelled, skiprows=skiprows, nrows=nrows)
        self.yelp_labelled = pd.read_csv(yelp_labelled, skiprows=skiprows, nrows=nrows)
        if nrows == None:
            self.rows = 3000
        else:
            self.rows = 3*nrows

    def __len__(self):
        return len(self.amazon_cells_labelled) + len(self.imdb_labelled) + len(self.yelp_labelled)

    def __getitem__(self, idx): #use https://stackoverflow.com/questions/55801167/how-to-read-a-csv-to-pandas-and-get-the-value-of-one-cell
        if 0 <= idx and idx <= (self.rows)/3 - 1:
            index = int(idx)
            return self.amazon_cells_labelled.values[index][0], self.amazon_cells_labelled.values[index][1]
        elif (self.rows)/3 <= idx and idx <= 2*(self.rows)/3 - 1:
            # print(f"idx - self.rows/3 is {idx - self.rows/3}")
            index = int(idx - self.rows/3)
            return self.imdb_labelled.values[index][0], self.imdb_labelled.values[index][1]
        elif 2*(self.rows)/3 <= idx and idx <= (self.rows) - 1:
            index = int(idx - 2*self.rows/3)
            # print(f"idx - 2*self.rows/3 is {idx - 2*self.rows/3}")
            return self.yelp_labelled.values[index][0], self.yelp_labelled.values[index][1]
        else:
            pass

#need to split into training and test data, probably 2400 and 600 items each unless I can get a larger dataset

amazonFile = "/home/sahireellahy/Personal Stuffs/UCLA/Michael Murray Research Project/sentiment labelled sentences/amazon_cells_labelled.csv"
imdbFile = "/home/sahireellahy/Personal Stuffs/UCLA/Michael Murray Research Project/sentiment labelled sentences/imdb_labelled.csv"
yelpFile = "/home/sahireellahy/Personal Stuffs/UCLA/Michael Murray Research Project/sentiment labelled sentences/yelp_labelled.csv"

training_data = SentimentSentencesDataset(amazonFile, imdbFile, yelpFile, nrows=800) #make instance of SentimentSentenceDataset
test_data = SentimentSentencesDataset(amazonFile, imdbFile, yelpFile, skiprows=800 , nrows=200) #make instance of SentimentSentenceDataset

batch_size = 50
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True) #now, unmuck THIS; getting index out of bound errors
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True) #now, unmuck THIS; getting index out of bound errors

# Display sentence and label.
train_sentences, train_labels = next(iter(train_dataloader)) #stores a batch of sentences and labels in train_features and train_labels respectively

print(f"Sentence batch shape: {len(train_sentences)}")
print(f"Labels batch shape: {len(train_labels)}")
print()

sampleSentence, sampleLabel = training_data.__getitem__(idx=0)
print(f"An amazon review sentence is \"{sampleSentence}\" and it's label is {sampleLabel}")

sampleSentence, sampleLabel = training_data.__getitem__(idx=1000)
print(f"An imdb review sentence is \"{sampleSentence}\" and it's label is {sampleLabel}")

sampleSentence, sampleLabel = training_data.__getitem__(idx=2000)
print(f"A yelp review sentence is \"{sampleSentence}\" and it's label is {sampleLabel}")
print()


# (6) use GPU (TODO: need to put above)
#we will transfer the NN onto the GPU
#define our device as the first visible cuda device is one is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device is {device}")

#send parameters and buffers to CUDA tensors
# model.to(device)

#send inputs and targets at each step to GPU too
# inputs, labels = data[0].to(device), data[1].to(device)


# define network
# need encoder and decoder, attentionlayer, and some basic FC layers

import torch.nn.functional as Functional
# import torchtext
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
#consider using https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76 to work out embedding nonsense
#will use 50d GloVe vectors from Wikipedia 2014 + Gigaword 5 corpora found here https://github.com/stanfordnlp/GloVe
#don't think I need embeddings since I have the above word vectors

tokenizer = get_tokenizer("basic_english") #documentation at https://pytorch.org/text/stable/data_utils.html
global_vectors = GloVe(name='6B', dim=50)


# "The food was delicious!" -> [the, food, was, delicious, !] ->

torch.set_grad_enabled(True)  # Context-manager; try to address RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn

class AttentionNet(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.selfAttBlocks = blocks
        QKV_size = 50 + 1 #50 depends on length of GloVe vectors
        self.attention = attentionlayer.MyAttentionLayer(QKV_size)
        # self.attention = nn.MultiheadAttention(QKV_size, 1)
        self.fc1 = nn.Linear(51, 27)
        self.fc2 = nn.Linear(27, 18)
        self.fc3 = nn.Linear(18, 2)


    def forward(self, x):
        print(f"x is size {x.size()}")
        for i in range(0,self.selfAttBlocks):
            x = self.attention(x) #my attention
        print(f"x is {x.size()}")
        x = Functional.relu(self.fc1(x))
        x = Functional.relu(self.fc2(x))
        x = self.fc3(x)
        softmax = nn.Softmax(dim=0)
        x = softmax(x)
        # x = torch.argmax(x) #seems to be nondiff and commenting seems to fix "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
        return x

def preprocess(sentence):
    sentence = tokenizer(sentence) #split up tokens in sentence
    sentence = global_vectors.get_vecs_by_tokens(sentence, lower_case_backup=True) #torch of tokens x dim of GloVe vectors
    length = sentence.size(dim=0) #number of tokens
    position_vec = torch.zeros((length, 1)) #vec of positional encodings to append to embeddings; 50 depends on length of GloVe vectors
    for idx in range(0, position_vec.size(dim=0)):
        position_vec[idx, 0] = idx + 1 #add positional encodings
    sentence = torch.cat((sentence, position_vec), dim=1)
    return sentence




model = AttentionNet(1) #define what the model is
model.to(device) #send parameters and buffers to CUDA tensors

#example sentence, but some weird stuff with cuda
exampleSentence = "I don't like meat."
# exampleSentence.to(device)
preprocessedSent = preprocess(exampleSentence)
preprocessedSent = preprocessedSent.to(device)
# print(f"the preprocessed sentence is on the GPU: {preprocessedSent.is_cuda}")
example = torch.argmax(model(preprocessedSent))
print(f"Let's see what model says about \"I don't like meat.\": {example}")




# print(f"model is on cuda: {model.is_cuda}")
# model.to(device) #send parameters and buffers to CUDA tensors


# comment this out for now, will uncomment once I figure out how to loop
# training loop with tensorboard training graph
writer = SummaryWriter()

# model = AttentionNet() #define what the model is
criterion = nn.MSELoss() #choose the loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #choose how to adjust parameters


"""
def train_model(iter): #iter is how many times to go through entire training dataset. Maybe use https://datagy.io/pytorch-dataloader/
    for epoch in range(iter):
        y1 = model(inputOfModel) #inputOfModel is what would be input to model. This var would loop through things in dataset. Got to FIX this
        loss = criterion(y1, y) #y1 is the model's prediction and y is the actual label
        writer.add_scalar("Loss/train", loss, epoch) #add an entry between loss and epochs where loss is the "tag" and epoch is the "scalar_value"
        optimizer.zero_grad() #zero out gradient
        loss.backward() #compute gradient
        optimizer.step() #adjust parameters

train_model(10) #go through dataset 10 times
writer.flush() #makes sure pending events are written to disk (what does this mean?)

writer.close() #done with summary writer



# example of input and output, print overall accuracy and accuracy in each label


 """

torch.set_grad_enabled(True)
#figure out what's happening here and use it to fix above
#(4) 
#loop over data iterator and feed inputs to the network and optimize
def train_model(iter):
    loss = 0.0
    for epoch in range(iter):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # writer.add_scalar("Loss/train", loss, epoch) #add an entry between loss and epochs where loss is the "tag" and epoch is the "scalar_value"
            loss = 0.0
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # inputs = inputs.to(device)
            # labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            for idx, input in enumerate(inputs, 0):
                preprocessedInput = preprocess(input)
                preprocessedInput = preprocessedInput.to(device)
                # print(f"preprocessedInput is on cuda: {preprocessedInput.is_cuda}")
                output = model(preprocessedInput) #can be done better by preprocessing input outside of this loop and feeding all inputs to model at once with model(inputs)
                output = output.to(torch.float)
                labelvect = torch.zeros(2)
                labelvect[labels[idx]] = 1
                labelvect = labelvect.to(device)
                loss += criterion(output, labelvect)
                # loss.backward()
            # loss = criterion(outputs, labels)
            # loss = criterion(output, labels)
            # print(f"I will add an entry to tensorboard with loss {loss} and epoch {epoch}.")
            writer.add_scalar("Loss/train", loss, epoch) #add an entry between loss and epochs where loss is the "tag" and epoch is the "scalar_value"
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:    # print every 5 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.3f}')
                running_loss = 0.0


model.load_state_dict(torch.load('./Attention_net.pth'))
model.to(device)
epochs = 100
train_model(epochs) #train the model here
writer.flush() #makes sure pending events are written to disk (what does this mean?)
writer.close() #done with summary writer

# run "tensorboard --logdir=runs" to see output
# I don't understand graph, seems to be epoch on X axis and loss on Y axis, but the loss values on graph don't match console output. Even putting writer.add_scalar(...) before loss = 0.0 doesn't

#IT RUNS!!!



# (1) save and load trained model
PATH = './Attention_net.pth'
torch.save(model.state_dict(), PATH)


model.eval() #set model to evaluation mode
# (2) test trained network on test data (requires fixing dataloaders)
correct = 0
total = 0
# not training, so no need to calculate gradients for outputs
with torch.no_grad():
    for data in test_dataloader:
        sentences, labels = data # calculate outputs by running images through the network
        # print(f"len of labels is {len(labels)}")
        # outputs = model(sentences)
        outputs = torch.empty(50, 2)
        for idx, sentence in enumerate(sentences, 0):
            preprocessedSent = preprocess(sentence)
            preprocessedSent = preprocessedSent.to(device)
            outputs[idx] = model(preprocessedSent)
        # the class with the highest energy is what we choose as prediction
        # _, predicted = torch.max(outputs.data, 1)
        # predicted = torch.argmax(outputs, 1)
        predicted = torch.argmax(outputs, 1)
        # print(f"predicted is {predicted}")
        # print(f"predicted == labels is {predicted == labels}")
        # print(f"True + False = {True + False}")
        total += len(labels)
        correct += (predicted == labels).sum().item()
        # print(f"total is {total} and correct is {correct}")

print(f'Accuracy of the network on the 600 test images: {100 * (correct / total)} %')

# (3) give example of model's input and output


# (4) check how network performs on whole dataset
classes = ("positive", "negative")
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# (5) check performance per class

# again no gradients needed
with torch.no_grad():
    for data in test_dataloader:
        sentences, labels = data
        outputs = torch.empty(50, 2)
        for idx, sentence in enumerate(sentences, 0):
            preprocessedSent = preprocess(sentence)
            preprocessedSent = preprocessedSent.to(device)
            outputs[idx] = model(preprocessedSent)
        # print(f"outputs is {outputs}")
        predictions = torch.argmax(outputs, 1)
        # print(f"predictions is {predictions}")
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            # print(f"label is {label} and prediction is {prediction}")
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')




#TODO: (1) figure out what's happening with Tensorboard, (2) get pause and resume working training, (3) add more attention layers