# Assignment 11 - Python

## Directions
Learn a word2vec model from fake news dataset and a real news dataset. We 
will use the word2vec model implemented in the Python Gensim library. Use
a hidden layer of 300 and 1000 nodes separately to represent each word 
in the word2vec training. Use two different window sizes of 2 and 5. 

You can set workers to 10 and train it faster on multiple cores. In order to
do this you have to set tasks-per-node to 10 like below:

#SBATCH --tasks-per-node=10

Now we have two sets of word representations learnt from different datasets. 
Output the top 5 most similar words to the following ones from each 
representation.

1. Hillary
2. Trump
3. Obama
4. Immigration

In order to do this we first normalize all vector representations (set them 
to Euclidean length 1). Consider the vector x for a given word w. We 
compare the cosine similarity between x and the vectors x' for each word w' 
in the fake news dataset first. We then output the top 5 words with highest 
similarity. We then do the same for the real news and then see if the top 
similar words differ considerably.

Submit your assignments as two files train.py and test.py. Make
train.py take two inputs: the text dataset on which to learn the words and
a model file name to save the word2vec model to.

python train.py <text data> <word2vec model file to save the model> 

Make test.py take three inputs: text dataset, word2vec model, a query file 
containing five query words. The output should be the top five most similar 
words to each word in the query file.

python test.py <text data> <word2vec model file from train.py> <query words filename>
