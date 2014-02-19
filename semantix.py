# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:09:52 2014

@author: xander
"""
import re
from nltk.probability import FreqDist
from scipy.sparse import lil_matrix
import divisi2
from numpy import log as npl


# open the estonian open subtitles corpus
# strip non-alpha characters
# set case to lower

infile = open("walden.txt", "r")
raw = infile.read()
infile.close()

# define parameters
windowSize = 1
Nwords = 4000
Ncontexts = 200

print "Processing text..."
# terms = columns, contexts = rows (dimensions of the matrix)
words = [w.lower() for w in raw.split() if re.match(ur"^[^\W\d_]+$", w)][:100000]
vocab = FreqDist(words)
terms = vocab.keys()[:Nwords]
contexts = vocab.keys()[:Ncontexts]
N = len(words)

print "Creating count matrix..."
# create sparse matrix
m = lil_matrix((Nwords, Ncontexts))
for i in range(2, N-2):
    # progress check
    if i % 1000 == 0:
        print i
    # loop through windows, add to matrix
    if words[i] in terms:
        indices = [i-2,i-1,i+1,i+2]
        for j in indices:
            if words[j] in contexts:
                m[terms.index(words[i]),contexts.index(words[j])] += 1.

print "== Generating +PMI =="
print "Normalizing..."

m = m/m.sum()
column_sum = m.sum(0)
row_sum = m.sum(1)
denominator = lil_matrix(row_sum.dot(column_sum))
m = m/denominator

print "Getting log..."
m[m!=0] = npl(m[m!=0])
m[m < 0] = 0.

print " == SVD =="
divisi_sparse = divisi2.SparseMatrix(m.A, row_labels=terms, col_labels=contexts)
U, S, V = divisi_sparse.svd(k=100)

# two, three million in english
# compute nearest neighbors or cosine distance
# magnus sahlgren