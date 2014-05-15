import re
from numpy import save as npsave
from numpy import log as numpy_log
from nltk.probability import FreqDist
from scipy.sparse import lil_matrix

'''
Here is long, sort of obtuse script I wrote. If you define these paths at the beginning, you'll end up with
1. A matrix of PPMI values (word-context co-occurrences)
2. A matrix of lemmatized values (word-context co-occurrences)
3. A matrix of morphological information (word-case co-occurrences)
We're using tree-tagger to lemmatize.
'''

en_noun_file = open("path", "r")
en_raw_file_path = "path"
en_treetagger_file_path = "path"

et_noun_file = open("path", "r")
et_raw_file_path = "path"
et_treetagger_file_path = "path"


# Two helper functions for splitting
def chopp(x):
	return x.split('+')[0]

def chope(x):
	return x.split('=')[0]

# List of nouns 
en_noun_pairs = [p.split() for p in en_noun_file.readlines()]
en_noun_file.seek(0)
en_noun_list = en_noun_file.read().split()

# RAW ###############################################################
en_tokens = []
print "Cleaning & tokenizing text..."
with open(en_raw_file_path, "r") as f:
	for line in f:
		split_line = line.split()
		for word in split_line:
			if re.match(ur"^[^\W\d_]+$", word, re.UNICODE):
				en_tokens.append(word)

window_size = 1
columns = 2000
rows = len(en_noun_list)
en_vocabulary = FreqDist(en_tokens)
en_contexts = en_vocabulary.keys()[:columns] 
en_corpus_length = len(en_tokens)
percentage = en_corpus_length / 10
percent_counter = 0

print "Creating the raw English matrix...\n"
en_matrix = lil_matrix((rows, columns))
for i in range(2, en_corpus_length-2):
    # this is just for a progress check
    if i % percentage == 0:
        percent_counter += 10
        print percent_counter, "%"

    # here is where we actually loop through everything
    if en_tokens[i] in en_noun_list:
        indicies = [i-window_size, i+window_size]
        for j in indicies:
            if en_tokens[j] in en_contexts:
                en_matrix[en_noun_list.index(en_tokens[i]), en_contexts.index(en_tokens[j])]+=1

print "\n== Generating +PMI values =="
en_matrix = en_matrix/en_matrix.sum()                                    # 1
print "1 - Sum"
en_matrix = en_matrix/lil_matrix(en_matrix.sum(1).dot(en_matrix.sum(0)))    # 2
print "2 - Normalize"
en_matrix[en_matrix != 0] = numpy_log(en_matrix[en_matrix!=0])              # 3
print "3 - Log"
en_matrix[en_matrix < 0] = 0.                                         # 4
npsave(open('en_raw_matrix.npy', "w"), en_matrix)
print "English raw matrix done!"


# MOR ###############################################################
en_cases = ['NN', 'NNS']
en_tokens = []
en_morph_matrix = lil_matrix((len(en_noun_list), len(en_cases)))

print "Processing text & making morphological matrix..."
with open(en_treetagger_file_path, "r") as f:
	for line in f:
		split_line = line.split()
		lemma = chopp(split_line[2])
		case = split_line[1]

		if lemma in en_noun_list:
			if case in en_cases:
				en_morph_matrix[en_noun_list.index(lemma), en_cases.index(case)] += 1
		
		if re.match(ur"^[^\W\d_]+$", lemma, re.UNICODE):
			en_tokens.append(lemma)

npsave(open("en_morph_matrix.npy", "w"), en_morph_matrix)
print "English morphological matrix done!"

# LEM ###############################################################

en_vocabulary = FreqDist(en_tokens)
en_contexts = en_vocabulary.keys()[:columns] 
en_corpus_length = len(en_tokens)
percentage = en_corpus_length / 10
percent_counter = 0

print "Creating the lemmatized English matrix...\n"
en_matrix = lil_matrix((rows, columns))
for i in range(2, en_corpus_length-2):
    # this is just for a progress check
    if i % percentage == 0:
        percent_counter += 10
        print percent_counter, "%"

    # here is where we actually loop through everything
    if en_tokens[i] in en_noun_list:
        indicies = [i-window_size, i+window_size]
        for j in indicies:
            if en_tokens[j] in en_contexts:
                en_matrix[en_noun_list.index(en_tokens[i]), en_contexts.index(en_tokens[j])]+=1

print "\n== Generating +PMI values =="
en_matrix = en_matrix/en_matrix.sum()                                    # 1
print "1 - Sum"
en_matrix = en_matrix/lil_matrix(en_matrix.sum(1).dot(en_matrix.sum(0)))    # 2
print "2 - Normalize"
en_matrix[en_matrix != 0] = numpy_log(en_matrix[en_matrix!=0])              # 3
print "3 - Log"
en_matrix[en_matrix < 0] = 0.                                         # 4
npsave(open('en_lem_matrix.npy', "w"), en_matrix)
print "English lemmatized matrix done!"

print "#" * 100
print "Estonian"
print "#" * 100

# Nouns
et_noun_pairs = [p.split() for p in et_noun_file.readlines()]
et_noun_file.seek(0)
et_noun_list = et_noun_file.read().split()

# RAW ###############################################################
et_tokens = []
print "Cleaning & tokenizing text..."
with open(et_raw_file_path, "r") as f:
	for line in f:
		split_line = line.split()
		for word in split_line:
			if re.match(ur"^[^\W\d_]+$", word, re.UNICODE):
				et_tokens.append(word)

window_size = 1
columns = 2000
rows = len(et_noun_list)
et_vocabulary = FreqDist(et_tokens)
et_contexts = et_vocabulary.keys()[:columns] 
et_corpus_length = len(et_tokens)
percentage = et_corpus_length / 10
percent_counter = 0

print "Creating the raw Estonian matrix...\n"
et_matrix = lil_matrix((rows, columns))
for i in range(2, et_corpus_length-2):
    # this is just for a progress check
    if i % percentage == 0:
        percent_counter += 10
        print percent_counter, "%"

    # here is where we actually loop through everything
    if et_tokens[i] in et_noun_list:
        indicies = [i-window_size, i+window_size]
        for j in indicies:
            if et_tokens[j] in et_contexts:
                et_matrix[et_noun_list.index(et_tokens[i]), et_contexts.index(et_tokens[j])]+=1

print "\n== Generating +PMI values =="
et_matrix = et_matrix/et_matrix.sum()                                    # 1
print "1 - Sum"
et_matrix = et_matrix/lil_matrix(et_matrix.sum(1).dot(et_matrix.sum(0)))    # 2
print "2 - Normalize"
et_matrix[et_matrix != 0] = numpy_log(et_matrix[et_matrix!=0])              # 3
print "3 - Log"
et_matrix[et_matrix < 0] = 0.                                         # 4
npsave(open('et_raw_matrix.npy', "w"), et_matrix)
print "Estonian raw matrix done!"


# MOR ###############################################################
# Cases
et_cases =['S.com','S.com.sg.nom','S.com.sg.gen','S.com.sg.part','S.com.sg.ill','S.com.sg.in','S.com.sg.el','S.com.sg.all','S.com.sg.ad','S.com.sg.abl','S.com.sg.tr','S.com.sg.term','S.com.sg.es','S.com.sg.abes','S.com.sg.kom','S.com.sg.adit','S.com.pl.nom','S.com.pl.gen','S.com.pl.part','S.com.pl.ill','S.com.pl.in','S.com.pl.el','S.com.pl.all','S.com.pl.ad','S.com.pl.abl','S.com.pl.tr','S.com.pl.term','S.com.pl.es','S.com.pl.abes','S.com.pl.kom']

# Tokens
et_tokens = []
et_morph_matrix = lil_matrix((len(et_noun_list), len(et_cases)))

print "Processing text & making morphological matrix..."
with open(et_treetagger_file_path, "r") as f:
	for line in f:
		split_line = line.split()
		lemma = chopp(split_line[2])
		if '=' in lemma:
			lemma = chope(split_line[2])
		case = split_line[1]

		if lemma in et_noun_list:
			if case in et_cases:
				et_morph_matrix[et_noun_list.index(lemma), et_cases.index(case)] += 1
		

		if re.match(ur"^[^\W\d_]+$", lemma, re.UNICODE):
			et_tokens.append(lemma)

npsave(open("et_morph_matrix.npy", "w"), et_morph_matrix)
print "Estonian morphological matrix done!"

# LEM ###############################################################

rows = len(et_noun_list)
et_vocabulary = FreqDist(et_tokens)
et_contexts = et_vocabulary.keys()[:columns] 
et_corpus_length = len(et_tokens)
percentage = et_corpus_length / 10
percent_counter = 0

print "Creating the Estonian lemmatized matrix...\n"
et_matrix = lil_matrix((rows, columns))
for i in range(2, et_corpus_length-2):
    # this is just for a progress check
    if i % percentage == 0:
        percent_counter += 10
        print percent_counter, "%"

    # here is where we actually loop through everything
    if et_tokens[i] in et_noun_list:
        indicies = [i-window_size, i+window_size]
        for j in indicies:
            if et_tokens[j] in et_contexts:
                et_matrix[et_noun_list.index(et_tokens[i]), et_contexts.index(et_tokens[j])]+=1

print "\n== Generating +PMI values =="
et_matrix = et_matrix/et_matrix.sum()                                    # 1
print "1 - Sum"
et_matrix = et_matrix/lil_matrix(et_matrix.sum(1).dot(et_matrix.sum(0)))    # 2
print "2 - Normalize"
et_matrix[et_matrix != 0] = numpy_log(et_matrix[et_matrix!=0])              # 3
print "3 - Log"
et_matrix[et_matrix < 0] = 0.                                         # 4
npsave(open('et_lem_matrix.npy', "w"), et_matrix)
print "Estonian lem matrix done!"
