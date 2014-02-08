"""
term-term semantic relationships from co-occurance statistics

@author: alexander johnson
"""

from __future__ import division
from nltk.corpus.reader.xmldocs import XMLCorpusReader as XML
from sklearn.metrics import pairwise_distances

from scipy import sparse
from numpy import matrix


# Load the EUbookshop corpus for Maltese
corpus_root = "mt"
mt = XML(corpus_root, ".*.xml")

fileids = [file for file in mt.fileids()]
files = []
for file in fileids[:2]:
    print '===== Processing ' + file + ' ====='
    files.append([w.lower().encode('UTF-8') for w in mt.words(file)])
    print 'Processed ' + file
print "~~~~~~~~~" * 5

# TEST
test = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty']


def create_window(words, windowsize):

    print 'Creating Window of size', windowsize, '...\n'

    # Container & Index
    window = []
    wordindex = 0
    percent = 10

    for w in set(words):

        if percent < ((wordindex / len(words)) * 100):
            print ((wordindex / len(words)) * 100), '%'
            percent += 10

        # each word needs a window & contexts
        term = [w]
        after = []
        before = []

        # iterate though the indicies
        for i in range(wordindex-1, wordindex-1-windowsize, -1):

            # index must be above zero or else we the end of the text
            if i >= 0:
                if i != wordindex:
                    before.append(words[i])

        # iterate thorough indicies
        for i in range(wordindex+1, wordindex+1 + windowsize):

            #try / except block to deal with indicies out of range
            try:
                if i != wordindex:
                    after.append(words[i])
            except IndexError:
                continue

        # append lists to term and term to window
        term.append(before)
        term.append(after)
        window.append(term)

        # raise index count
        # note: this normally wouldn't need to exist but since
        # words.index(w) wil count the ~first occurance of an item
        # we need to create our own counter
        wordindex += 1

    return window


def get_word_index(window):
    vectorindex = {}
    count = 0
    for word in window:
        vectorindex[word] = count
        count += 1
    return vectorindex


def tt_dictionary(words, window_size):

    print '\nCreating Term-Term Dictionary...\n'
    window = create_window(words, window_size)
    tt = {}

    # loop through all of the words
    for value1 in window:
        term1 = value1[0]

        # loop again for the second context
        for value2 in window:
            term2 = value2[0]

            count = 0

            # go through the before window and see if
            # there are matches between terms 1 & 2
            for potential in value1[1]:
                if potential == term2:
                    count += 1

            # go through the after window and see if
            # there are matches between terms 1 & 2
            for potential in value1[2]:
                if potential == term2:
                    count += 1

            # add to count, create key if it doesn't exist
            if (term1, term2) in tt.keys():
                tt[(term1, term2)] += count
            else:
                tt[(term1, term2)] = count

    return tt


def tt_matrix(dictionary):

    print '\nCreating Term-Term Matrix...\n'

    # this container holds the preprocessed values
    matrix_list = []

    count = 0
    percent = 10

    for key, value in dictionary.items():

        if percent < ((count / len(dictionary.items()) * 100)):
            print (count / len(dictionary.items()) * 100), '%'
            percent += 10

        #this container holds (context, count)s so we can sort
        prematrix = []
        for match, match_value in dictionary.items():
            if match[0] == key[0]:
                prematrix.append((match, match_value))
        prematrix.sort()
        matrix_list.append(prematrix)

        count += 1

    # this string will be the values of the matrix
    matrix_string = ''

    for term in matrix_list:

        # this is one row's data
        term_string = ' '
        for pair in term:
            term_string += str(pair[1]) + ' '
        term_string += ' ; '
        matrix_string += term_string

    # ending ; will break everything so take it out
    matrix_string = matrix_string[:-2]

    # sparse matrix choice: Compressed Sparse Row Matrix
    return sparse.csr_matrix(matrix(matrix_string))


def compute_pmi(sparse_matrix):
   
    # summation of each row & each column
    # divide each cell by the product of the two sums
    # take log of the cell
    # if its negative, it becomes zero 
     
    pass
    
    
# baby test

from nltk.corpus import gutenberg
from nltk.tokenize import wordpunct_tokenize
raw = gutenberg.raw('austen-emma.txt')
tokens = wordpunct_tokenize(raw[:500])
d = tt_dictionary(tokens, 2)
m = tt_matrix(d)



# i can try linalg.svd but its probably slow

# divisi - using C for SVD

# MDP - modular data processing toolkit for python
# algorithmnic complexity

# PMI & SVD is the goal right now

# test on 100K