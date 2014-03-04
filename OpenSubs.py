"""
Semantic Vectors, SVD,  Morphological Vectors, Oh My !

@author: xander johnson

"""
import re
import divisi2
from scipy.sparse import lil_matrix
from scipy.stats import pearsonr
from numpy import log as npl
from numpy import save 
from nltk.probability import FreqDist
from nltk.metrics.association import BigramAssocMeasures as BAM
from nltk.cluster.util import cosine_distance
from matplotlib import pyplot as plt

print "Processing text..."

# open up the OpenSubtitles2013 corpus (English) from
# http://opus.lingfil.uu.se/OpenSubtitles2013.php
open_subtitles_file = open("unpack/xaa", "r")
open_subtitles_raw = open_subtitles_file.read()
# use regex to strip punctuation
open_subs = [w.lower() for w in open_subtitles_raw.split() 
            if re.match(ur"^[^\W\d_]+$", w)]
open_subtitles_file.close()

# open up the "gold standard" for similarity comparisions
gold_file = open("MTURK-771.csv", "r")
gold_data = [line.strip().split(',') for line in gold_file.readlines()]
gold_file.close()

# define parameters
Nwords = 40000
Ncontexts = 2000

# terms = columns, contexts = rows (dimensions of the matrix)
words = open_subs
vocab = FreqDist(words)
terms = vocab.keys()[:Nwords]
contexts = vocab.keys()[:Ncontexts]
N = len(words)

print "Creating count matrix..."
# create sparse matrix
m = lil_matrix((Nwords, Ncontexts))
for i in range(2, N-2):
    # progress check
    if i % 100000 == 0:
        print i
    # loop through windows, add to matrix
    if words[i] in terms:
        indices = [i-1,i+1]
        for j in indices:
            if words[j] in contexts:
                m[terms.index(words[i]),contexts.index(words[j])] += 1.

c = m.copy()

# Calculate entire PPMI to compare to single PPMI
print "== Generating +PMI =="
print "Normalizing..."
m = m/m.sum()
column_sum = m.sum(0)
row_sum = m.sum(1)
denominator = lil_matrix(row_sum.dot(column_sum))
print "..."
m = m/denominator

print "Getting log..."
m[m!=0] = npl(m[m!=0])
m[m < 0] = 0.

#print " == SVD =="
#divisi_sparse = divisi2.SparseMatrix(m.A)
#U, S, V = divisi_sparse.svd(k=300)

semantic_vectors = m.A

# Get overlap between contexts & gold similarities
overlap = []
for g in gold_data:
    if g[0] in contexts:
        if g[1] in contexts:
            overlap.append(g)      

# Create a list of words & the two ratings
# Also - normalize the similarities and convert distance to similariy for cosines
values = []
for d in overlap:
    values.append([d[0], d[1], (float(d[2])-1)/4, 1-cosine_distance(semantic_vectors[terms.index(d[0])], semantic_vectors[terms.index(d[1])])])

_, _, sims, cosines = zip(*values)
pearsonr(sims, cosines)