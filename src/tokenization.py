import pandas as pd
import numpy as np


#  sample string
print()
print('[INFO] sample string.')
sample_string = 'Since 1995, Issa was declared as the richest man on earth.'
print(sample_string)


# The simplest way to tokenize a sentence is to use a whitespace within a string as a delimiter of words

'''
    A major draw back of this method is it include the ending punctuation mark which actually
    needs to be separated
'''
print()
print('[INFO] tokenization with str.split tool.')
tokens = str.split(sample_string)
print(tokens)

# creating a vocabulary list, sorted lexically
print()
print('[INFO] vocabulary list sorted lexically')
vocabulary = sorted(set(tokens))
print(' '.join(vocabulary))

# one_hot_encoding
print('[INFO] one hot encoding')
token_size = len(tokens)
vocab_size = len(vocabulary)
one_hot_vector = np.zeros((token_size, vocab_size), int)

# loop over the tokens to update the word indices
for (i, token) in enumerate(tokens):
    one_hot_vector[i, vocabulary.index(token)] = 1

# create a dataframe to visualize one hot encoding
data = pd.DataFrame(one_hot_vector, columns=vocabulary)
print(data)

'''
    A major drawback of one hot encoding is that it is subjected to bigger matrices i.e. Bigger memory size
    To resolve that the sentences or documents can be presented in a bag-of-words vector useful for
    summarizing the essence of a document. Below is how you can put the tokens into a binary vector
    indicating the presence or the absence of a particular word in a particular sentence
'''

# create a sentence bow
sentence_bow = {}

# loop over the tokens and insert tokens with their index into a sentence bow
print()
print('[INFO] bag of words block')
for token in tokens:
    sentence_bow[token] = 1

# note while using sorted() method, items are arranged in the same way as ASCII and Unicode character sets are arranged
print(sorted(sentence_bow.items()))
print()

# present bag of words into a pandas dataframe
bag_of_word = pd.DataFrame(pd.Series(sentence_bow), columns=['sent'])
print(bag_of_word.T)

# Add more sentences
sentences = 'Since 1995, Issa was declared as the richest man on the earth.\n'
sentences += 'Issa is a fifth child in a family of six children.\n'
sentences += 'Before business he first went through education pipeline and acquire his first degree at the age of 25.\n'
sentences += 'At the age of 26 he started his own venture on technology which went by the name RadonPlus.'

# create joint tokens
print()
sample_sentences = sentences.split('\n')

# create corpus (holds a tokens for a single sentence)
corpus = {}

for i, sent in enumerate(sample_sentences):
    corpus['sent{}'.format(i)] = dict((token, 1) for token in sorted(sent.split()))

# store corpus in a pandas dataframe
data = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T

print(data[data.columns[:11]])

'''
    From here you can notice the word overlaps among sentences
    From there we can compute these overlaps whenever we want to compare documents or search for similar documents
    One way to chek for the similarities between sentences is to count the number of overlapping tokens using 
    dot product
'''

# First transpose a dataframe to be aligned as default style
new_data = data.T

# measure ovelaps
overlap0 = new_data.sent0.dot(new_data.sent1)
overlap1 = new_data.sent0.dot(new_data.sent2)
overlap2 = new_data.sent0.dot(new_data.sent3)
print()
print('[INFO] sentence 0 vs 1 overlaps: {}'.format(overlap0))
print('[INFO] sentence 0 vs 2 overlaps: {}'.format(overlap1))
print('[INFO] sentence 0 vs 3 overlaps: {}'.format(overlap2))

# find the actual overlapping words
overlap = {}

for k, v in (new_data.sent0 & new_data.sent3).items():
    if v:
        overlap[k] = v

print()
print(overlap)
