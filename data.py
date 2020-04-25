import torch 
import logging
from pytorch_transformers import BertModel, OpenAIGPTModel, GPT2Model
import nltk
from tqdm import tqdm
import copy
import pickle
nltk.download('punkt')
import time
import sklearn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from gensim.models import KeyedVectors
import io
from tokenization_bert import BertTokenizer, BasicTokenizer
from tokenization_gpt2 import GPT2Tokenizer


translation_dict = {'misspend' : 'waste', 'wandmaker' : 'magician', 'bilstm' : 'machine', 'brklyn' : 'brooklyn', 'theatre' : 'theater', 'harbour' : 'harbor', 'arafat' : 'chairman', 'colour' : 'color', 'grey': 'gray', 'ruiz' : 'gomez'}


def fetch_word2vec_embeddings(path_prefix, vocab, pickled=False):
	print("Fetching Word2vec Embeddings")
	pickle_path = path_prefix + "word2vec.vocab_size={}.pickle".format(len(vocab))
	if pickled:
		output = pickle.load(open(pickle_path, 'rb'))
	else:
		filepath = "GoogleNews-vectors-negative300.bin"
		wv_from_bin = KeyedVectors.load_word2vec_format(path_prefix + filepath, binary=True)
		output = {w : wv_from_bin[translation_dict.get(w, w)] for w in vocab}
		pickle.dump(output, open(pickle_path, 'wb'))
	return output


def fetch_glove_embeddings(path_prefix, pickled=False):
	print("Fetching GloVe Embeddings")
	dimension = 300
	print("Fetching GloVe {} Dimensional Embeddings".format(dimension))
	path = path_prefix + "glove.6B.{}d".format(dimension)
	pickle_path = path + ".pickle"
	if pickled:
		output = pickle.load(open(pickle_path, 'rb'))
		embedding_matrix, _, _ = output
	else:
		w2i, i2w = {}, {}
		embedding_matrix = []
		with open(path + ".txt", mode = 'r') as f:
			for i, line in tqdm(enumerate(f)):
				tokens = line.split()
				word = tokens[0]
				embedding = np.array([float(val) for val in tokens[1:]] )
				w2i[word] = i 
				i2w[i] = word 
				embedding_matrix.append(embedding)
				assert len(embedding) == dimension
		embedding_matrix = np.array(embedding_matrix)
		output = (embedding_matrix, w2i, i2w)
		pickle.dump(output, open(pickle_path, 'wb'))
	print("Shape of GloVe Embedding Matrix: {}".format(embedding_matrix.shape))
	return output


def fetch_sents_from_uniform(data, vocab, k):
	output = []
	words = sorted(vocab)
	print([(t, len(data[t])) for t in sorted(data, key = lambda x: len(data[x]))][:20])
	for w in words:
		output.extend(data[w][:k])
	print("Ideal dataset length: {}, dataset length: {}".format(len(vocab) * k, len(output)))
	return output


def fetch_weighted_dataset_initially(path_prefix, f_name, weights, N, vocab, pickled=False):
	basic_tokenizer = BasicTokenizer()
	V = len(vocab)
	assert len(weights) == V
	count_list = [round(t * N) for t in weights]
	counts = {w : count_list[i] for i,w in enumerate(sorted(vocab))}
	print("Fetching {} sentences weighted from {} with a filter vocabulary size of {}".format(N, f_name, len(vocab)))
	f_name = path_prefix + f_name
	pickle_path = f_name + "{}_weighted_sents.filter_size={}.pickle".format(N, len(vocab))
	examples = {w : counts[w] for w in vocab}
	data = {w : [] for w in vocab}
	if pickled:
		data = pickle.load(open(pickle_path, 'rb'))
	else:
		with open(f_name, mode = 'r') as f:
			for i, row in tqdm(enumerate(f)):
				index, text = row.split(',', 1)
				sent_text = nltk.sent_tokenize(text)
				for s in sent_text:
					if 7 < len(nltk.word_tokenize(s)) < 75:
						for w in basic_tokenizer.tokenize(s):
							if w in examples and examples[w]:
								data[w].append(s)
								examples[w] -= 1
				if i % 100 == 0:
					print(sum(list(examples.values())), sum([len(data[k]) for k in data]) )
				if sum(examples.values()) < (0.01 * N):
					break
		pickle.dump(data, open(pickle_path, 'wb')) 
	print("Fetched Weighted Dataset of {} sentences for a {} size vocab".format(N, V))
	return data	


def fetch_uniform_dataset_initially(path_prefix, f_name, example_count, vocab, pickled=False, file_str=""):
	basic_tokenizer = BasicTokenizer()
	print("Fetching {} sentences per word from {} with a filter vocabulary size of {}".format(example_count, f_name, len(vocab)))
	f_name = path_prefix + f_name
	pickle_path = f_name + "{}.{}_uniform_nice_sents.filter_size={}.pickle".format(file_str, example_count, len(vocab))
	examples = {w : example_count for w in vocab}
	data = {w : [] for w in vocab}
	if pickled:
		data = pickle.load(open(pickle_path, 'rb'))
	else:
		with open(f_name, mode = 'r') as f:
			for i, row in tqdm(enumerate(f)):
				index, text = row.split(',', 1)
				sent_text = nltk.sent_tokenize(text)
				for s in sent_text:
					if 7 < len(nltk.word_tokenize(s)) < 75:
						for w in basic_tokenizer.tokenize(s):
							if w in examples and examples[w]:
								data[w].append(s)
								examples[w] -= 1
				if i % 100 == 0:
					if sum(list(examples.values())) == 0 or i > 50000:
						break
					print(sum(list(examples.values())), sum([len(data[k]) for k in data]))
					print(i, [k for k, v in examples.items() if v > 0])
		pickle.dump(data, open(pickle_path, 'wb')) 
	return data	


def fetch_pooling_dataset(path_prefix, f_name, n, pickled=False, filter_vocab=None):
	basic_tokenizer = BasicTokenizer()
	print("Fetching {} sentences from {} with a filter vocabulary size of {}".format(n, f_name, len(filter_vocab)))
	f_name = path_prefix + f_name
	if filter_vocab is None:
		pickle_path = f_name + "{}_sents.pickle".format(n)
	else:
		pickle_path = f_name + "{}_sents.filter_size={}.pickle".format(n, len(filter_vocab))
	data = []
	if pickled:
		data = pickle.load(open(pickle_path, 'rb'))
	else:
		with open(f_name, mode = 'r') as f:
			for i, row in tqdm(enumerate(f)):
					index, text = row.split(',', 1)
					sent_text = nltk.sent_tokenize(text)
					for s in sent_text:
						if 7 < len(nltk.word_tokenize(s)) < 75:
							if filter_vocab is not None:
								if any([w in filter_vocab for w in basic_tokenizer.tokenize(s)]):
									data.append(s)
							else:		
								data.append(s)
					if len(data) > n:
						data = data[:n]
						break
					if i % 100 == 0:
						print(i, len(data))
		pickle.dump(data, open(pickle_path, 'wb')) 
	return data


def fetch_uniform_dataset(f_name, sentences, n, filter_vocab, pickled=False):
	V = len(filter_vocab)
	n = V * (n // V)
	vocab = sorted(list(filter_vocab))
	basic_tokenizer = BasicTokenizer()
	print("Fetching {} sentences from {} with a filter vocabulary size of {}".format(n, f_name, V))
	pickle_path = f_name + "{}_sents.uniform.filter_size={}.pickle".format(n, V)
	data = []
	if pickled:
		data = pickle.load(open(pickle_path, 'rb'))
	else:
		for w in tqdm(vocab):
			i = 0
			for s in sentences:
				if i == n // V:
					break
				if w in basic_tokenizer.tokenize(s):
					data.append(s)
					i += 1
			print(w, len(data), n // V)
		pickle.dump(data, open(pickle_path, 'wb')) 
	return data


def fetch_wordsim_datasets(path_prefix):
	RG, WS, SL, SV, = [], [], [], []
	words = set()
	word_counts = {}
	print("Fetching RG65 Word Similarity Dataset")
	with open(path_prefix + "RG65.csv") as f:
		for line in f:
			w1, w2, score = line.split(";")
			score = float(score[:-3])
			w1, w2 = w1.lower(), w2.lower()
			RG.append((w1, w2, score))
			words.add(w1)
			words.add(w2)
			word_counts[w1] = word_counts.get(w1, 0) + 1
			word_counts[w2] = word_counts.get(w2, 0) + 1
	print("Fetching WS353 Word Similarity Dataset")
	with open(path_prefix + "WS353.csv") as f:
		skip_first = True
		for line in f:
			if skip_first:
				skip_first = False
			else:
				w1, w2, score = line.split(",")
				score = float(score[:-2])
				w1, w2 = w1.lower(), w2.lower()
				WS.append((w1, w2, score))
				words.add(w1)
				words.add(w2)
				word_counts[w1] = word_counts.get(w1, 0) + 1
				word_counts[w2] = word_counts.get(w2, 0) + 1
	print("Fetching SimLex999 Word Similarity Dataset")
	with open(path_prefix + "SimLex-999.txt") as f:
		next(f)
		for line in f:
			line = line.split()
			w1, w2, score = line[0].lower(), line[1].lower(), float(line[3])
			words.add(w1)
			words.add(w2)
			word_counts[w1] = word_counts.get(w1, 0) + 1
			word_counts[w2] = word_counts.get(w2, 0) + 1
			SL.append((w1, w2, score))
	print("Fetching SimVerb3500 Similarity Dataset")
	with open(path_prefix + "SimVerb-3500.txt") as f:
		for line in f:
			line = line.split()
			w1, w2, score = line[0].lower(), line[1].lower(), float(line[3])
			words.add(w1)
			words.add(w2)
			word_counts[w1] = word_counts.get(w1, 0) + 1
			word_counts[w2] = word_counts.get(w2, 0) + 1
			SV.append((w1, w2, score))	
	print("Vocabulary Size:", len(words))
	print("Words:" , sum(word_counts.values()))
	return RG, WS, SL, SV, words, word_counts


def fetch_static_embeddings(path_prefix, word2vec, glove, w2i, vocab, pickled=False):
	print("Fetching Word2vec and GloVe Static Embeddings")
	pickle_f = path_prefix + "static_vocab_size={}.decontextualized.pickle".format(len(vocab))
	if pickled:
		score_embeddings = pickle.load(open(pickle_f, 'rb'))
	else:
		score_embeddings = {}
		for w in vocab:
			if w in w2i:
				score_embeddings[w]["glove"] = glove[w2i[w]]
			else:
				print("Not in GloVe:", w)
				score_embeddings[w]["glove"] = glove[w2i[translation_dict[w]]]
			if w in word2vec:
				score_embeddings[w]["word2vec"] = word2vec[w]
			else:
				print("Not in Word2vec:", w)
		pickle.dump(score_embeddings, open(pickle_f, 'wb'))
	return score_embeddings


def fetch_professions(path_prefix):
	with open(path_prefix + 'professions.txt') as f:
		professions = [t for t in f]
	professions_str = professions[0].split(', [')
	professions = []
	for s in professions_str:
		if '"' in s:
			start = s.find('"')
			end = s.find('"', start + 1)
			profession = s[start + 1: end]
			if "_" not in profession:
				professions.append(profession)
	return professions
