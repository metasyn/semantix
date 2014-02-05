# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:53:40 2014

@author: xander
"""

from __future__ import division
from collections import OrderedDict
from nltk.corpus.reader.xmldocs import XMLCorpusReader as XML
from sklearn.metrics import pairwise_distances
import numpy as np



# Load the EUbookshop corpus for Maltese
corpus_root = "EUbookshop/mt"
mt = XML(corpus_root, ".*.xml")

fileids = [file for file in mt.fileids()]
files = []
for file in fileids[:2]:
    print '===== Processing ' + file + ' ====='
    files.append([word.lower().encode('UTF-8') for word in mt.words(file)])
    print 'Processed ' + file
print "~~~~~~~~~" * 5

# TEST
test = ['An', 'n-gram', 'tagger', 'is', 'a', 'generalization', 'of', 'a', 'unigram', 'tagger', 'whose', 'context', 'is', 'the', 'current', 'word', 'together', 'with', 'the', 'part-of-speech', 'tags', 'of', 'the', 'n-1', 'preceding', 'tokens,', 'as', 'shown', 'in', '5.9.', 'The', 'tag', 'to', 'be', 'chosen,', 'tn,', 'is', 'circled,', 'and', 'the', 'context', 'is', 'shaded', 'in', 'grey.', 'In', 'the', 'example', 'of', 'an', 'n-gram', 'tagger', 'shown', 'in', '5.9,', 'we', 'have', 'n=3;', 'that', 'is,', 'we', 'consider', 'the', 'tags', 'of', 'the', 'two', 'preceding', 'words', 'in', 'addition', 'to', 'the', 'current', 'word.', 'An', 'n-gram', 'tagger', 'picks', 'the', 'tag', 'that', 'is', 'most', 'likely', 'in', 'the', 'given', 'context.']


def createwindow(words, windowsize):
    
    # Container & Index
    window = []
    wordindex = 0
    
    for w in words:
        
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
        
        # append lists to term and term to windw
        term.append(before)
        term.append(after)
        window.append(term)
        
        # raise index count
        # note: this normaly wouldn't need to exist but since
        # words.index(w) wil count the ~first occurane of an item
        # we need to creae our own counter
        wordindex += 1
        
    return window
        
def ttdictionary(words, window_size):
    
    window = createwindow(words, window_size)
    tt = {}

    # loop through all of the words
    for value1 in window:
        term1 = value1[0]

        # loop again for the second context
        for value2 in window:
            term2 = value2[0]

            beforecount = 0
            aftercount = 0

            # go through the before window and see if
            # there are matches between terms 1 & 2

            for potential in value1[1]:
                if potential == term2:
                    beforecount += 1

            # add to count, create key if it doesn't exit
            if tt.has_key((term1, term2, 'before')):
                tt[(term1, term2, 'before')] += beforecount
            else:
                tt[(term1, term2, 'before')] = beforecount

            # go through the after window and see if
            # there are matches between terms 1 & 2
            for potential in value1[2]:
                if potential == term2:
                    aftercount += 1

            # add to count, create key if it doesn't exist
            if tt.has_key((term1, term2, 'after')):
                tt[(term1, term2, 'after')] += aftercount
            else:
                tt[(term1, term2, 'after')] = aftercount

    return tt

def ttmatrix(dictionary):
    matrixlist = []
    matrixstring = ''
    count = 0
    
    for k in dictionary.keys():
        prematrix = []     
        print count / len(dictionary.keys()), '%'
        
        # Get all keys of the same word
        for match, value in dictionary.items():
            if match[0] == k[0]:
                prematrix.append((match,value))
        

        prematrix.sort()
        matrixlist.append(prematrix)
        
        count +=1
        
    # Extract value from matrix list    
    for term in matrixlist:
        termstring = ''
        for context in term:
            termstring += str(context[1]) + ' '
            print context
        termstring += '; '
        print termstring
        matrixstring += termstring
    matrixstring = matrixstring[:-2]
    return np.matrix(matrixstring)


def ncs(matrix):
        ncs = 1-pairwise_distances(matrix, metric="cosine")
        return ncs


tt = ttdictionary(test, 4)
def mapncs(matrix, dictionary):
    pass

