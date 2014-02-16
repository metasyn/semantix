#from __future__ import division
import re
from nltk.probability import FreqDist
from scipy.sparse import lil_matrix
from numpy import log
from scipy.linalg import svd
from scipy.spatial.distance import cosine




# open the estonian open subtitles corpus
# strip non-alpha characters
# set case to lower

infile = open("500k.txt", "r")
raw = infile.read()
infile.close()

# define parameters
windowSize = 1
Nwords = 40000
Ncontext = 2000

# terms = columns, context = rows (dimensions of the matrix)
words = [w.lower() for w in raw.split() if re.match(ur"^[^\W\d_]+$", w)][:100000]
vocab = FreqDist(words)
terms = vocab.keys()[:Nwords]
context = vocab.keys()[:Ncontext]
N = len(words)

# create sparse matrix
m = lil_matrix((Nwords, Ncontext))

for i in range(2, N-2):
    # progress check
    if i % 10000 == 0:
        print i
    # loop through windows, add to matrix
    if words[i] in terms:
        indices = [i-2,i-1,i+1,i+2]
        for j in indices:
            if words[j] in context:
                m[terms.index(words[i]),context.index(words[j])] += 1.

# point wise mutual information
m = m/m.sum()
column_sum = m.sum(0)
row_sum = m.sum(1)
denominator = lil_matrix(row_sum.dot(column_sum))
m = m/denominator
m[m!=0] = log(m[m!=0])
# only need positive PMI 
m[m < 0] = 0.

# scipy svd
U, s, Vh = svd(m.A, full_matrices=False)


