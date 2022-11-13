# Review-Sentiment-Model
A (untrained) PyTorch neural Network that reads product reviews and classifies them as positive or negative. 
Created during a summer research project with Michael Murray as an exercise in PyTorch. Uses my custom self-attention module from the same project.
Gives tokens a positional encoding. Rather than using the Positional encoding used in Attention Is All You Need, I thought of appending an entry to each token corresponding to its percent position in a sentence. To illustrate this, consider a 9 word sentence ending in a period - the first word in the sentence gets 0.1 appended to its token, the second word gets 0.2 appended to its token,..., the ninth word gets 0.9 appended to its token, the period gets 1 appended to its token. 
It uses [a Kaggle] (https://www.kaggle.com/datasets/marklvl/sentiment-labelled-sentences-data-set) dataset. It uses 50 dimensional [GloVe embeddings] (https://github.com/stanfordnlp/GloVe) to tokenize sentences. You can view training curves with TensorBoard, you can save and load the model, and it runs on the GPU.