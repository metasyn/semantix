"""
Semantic Vectors Matrix Maker

* uses positive pointeise mutual information values 
* requires NLTK, Scipy & Numpy

@author: xander johnson
"""

import re
from scipy.sparse import lil_matrix
from numpy import log as numpy_log
from nltk.probability import FreqDist

def createTokensFromText(infile):
    """ Returns a tokenzied version of a text file (as a list)."""
    raw_text = infile.read()
    tokens = [word.lower() for word in raw_text.split() if
    re.match(ur"^[^\W\d_]+$", word, re.UNICODE)]
    return tokens 

def createMatrix(rows, columns, tokens, window_size=1):
    """
    The rows & columns should be numbers, which will represent the
    dimensions of the matrix. The tokens need to be seperated word tokens in a
    list format. This function will return a sparse matrix of +PMI values.

    The window_size is initialized at 1 for optimal results (see Bullinaria &
    Levy 2007 for an in-depth comparison of window size comparisons).
    """

    # We choose the most frequest tokens, then define the matrix dimensions.
    vocabulary = FreqDist(tokens)
    terms = vocabulary.keys()[:rows]
    contexts = vocabulary.keys()[:columns] 
    corpus_length = len(tokens)
    tenth_of_corpus = corpus_length / 10
    percent_counter = 0

    print "Creating the raw count matrix...\n"
    matrix = lil_matrix((rows, columns))
    for i in range(2, corpus_length-2):
        # this is just for a progress check
        if i % tenth_of_corpus == 0:
            percent_counter += 10
            print percent_counter, "%"

        # here is where we actually loop through everything
        if tokens[i] in terms:
            indicies = [i-window_size, i+window_size]
            for j in indicies:
                if tokens[j] in contexts:
                    matrix[terms.index(tokens[i]), contexts.index(tokens[j])]+=1
    
    # Now we want to calculate the positive pointwise mutual information from
    # the raw counts - PMI is defined as:
    # pmi(x; y) = log( p(x,y) / p(x)p(y)
    #
    # http://en.wikipedia.org/wiki/Pointwise_mutual_information
    #
    # 1. Normalize the values by diving the sum of the matrix.
    # 2. Divide by the dot product of row & column summations.
    # 3. Take the logarithm
    # 4. Set all negative numbers to zero to only retain positive values. 
    
    print "\n== Generating +PMI values =="
    matrix = matrix/matrix.sum()                                    # 1
    matrix = matrix/lil_matrix(matrix.sum(1).dot(matrix.sum(0)))    # 2
    matrix[matrix != 0] = numpy_log(matrix[matrix!=0])              # 3
    matrix[matrix < 0] = 0.                                         # 4

    return matrix

