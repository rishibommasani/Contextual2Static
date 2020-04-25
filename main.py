import torch 
import logging
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
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from scipy.stats.mstats import spearmanr
import io
import data
import scorer 
import bias
import distill
import garg_list
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

attributes = {'gender_pair' : bias.fetch_seed_pairs_vocab(), 'gender':bias.fetch_class_words(['male', 'female']), 'race':bias.fetch_class_words(['white', 'hispanic', 'asian']), 'religion':bias.fetch_class_words(['islam', 'christian'])}
bias2short = {'gender_pair' : 'gp', 'gender' : 'gen', 'race' : 'race', 'religion' : 'rel', 'adjectives' : 'adj', 'professions' : 'prof'}
neutrals = {'adjectives' : bias.fetch_adjectives(), 'professions' : bias.fetch_professions()}
dataset_lens = {'gender_pair' : 400, 'gender' : 794, 'race': 1120, 'religion' : 660, 'adjectives' : 8256, 'professions' : 5679}	
name2long = {'word2vec' : 'word2vec', 'GloVe': 'GloVe', 'bertsm' : 'BERT-12', 'gpt2sm' : 'GPT2-12', 'robertasm' : 'RoBERTa-12', 'xlnetsm' : 'XLNet-12', 'distilbertsm': 'DistilBERT-6', 'bertlg' : 'BERT-24',  'gpt2lg' : 'GPT2-24', 'robertalg' : 'RoBERTa-24', 'xlnetlg' : 'XLNet-24',}
server = False
path_prefix = ''

cuda = False


def fetch_weights(wc):
	total = sum(wc.values())
	return [wc[w] / total for w in sorted(wc.keys())]


def fix_backups(score_embeddings, backup_embeddings, vocab, embedding_keys, layers):
	backups = 0
	for word in vocab:
		if len(score_embeddings[word][0].keys()) == 0:
			backups += 1
			# print(word)
			for i in range(layers + 1):
				for macro, micro in embedding_keys:
					if macro in score_embeddings[word][i]:
						score_embeddings[word][i][macro][micro] = backup_embeddings[word][(i, micro)]
					else:
						score_embeddings[word][i][macro] = {micro: backup_embeddings[word][(i, micro)]}
	# print("Number of words requiring backups: {}".format(backups))
	return score_embeddings


def visualize(model_name, scores, decont, layers, legend_keys=None):
	if decont:
		if layers:
			x = list(range(layers + 1))
			legend_keys = ["vec_min", "vec_max", "vec_last", "vec_mean"]
			for task in scores:
				for legend_key in legend_keys:
					y = []
					for i in range(layers + 1):
						y.append(scores[task][(i, legend_key)])
					plt.plot(x, y)
				plt.title("Decontextualized {} Performance on {}".format(model_name, task))
				plt.legend(legend_keys)
				plt.show()
		else:
			N = len(scores) # number of tasks
			bar_keys = {k : [scores[t][k] for t in scores] for k in legend_keys}
			ind = np.arange(N) 
			width = 0.085
			for i, key in enumerate(legend_keys):
				plt.bar(ind + i * width, bar_keys[key], width, label=key)
			plt.title("Static Embedding Performance")
			plt.xticks(ind + width / 2, [t for t in scores]) 
			plt.legend(loc='best')
			plt.show()
	else:
		x = list(range(layers + 1))
		for task in scores:
			plt.title("Aggregated {} Performance on {}".format(model_name, task))
			for key in legend_keys:
				y = scores[task][key]
				plt.plot(x, y)
				plt.legend(legend_keys)
			plt.show()



def submission_visualize(model_name, cont_scores, decont_scores, layers, decont_embedding_keys, cont_embedding_keys):
	subword2label = {"vec_min" : "f = min; ", "vec_max" : "f = max; ", "vec_last" : "f = last; ", "vec_mean" : "f = mean; "}
	context2label = {"min" : "g = min", "max" : "g = max", "mean" : "g = mean", "decont" : "g = decont"}
	label2subword, label2context = {l : s for s, l in subword2label.items()}, {l : c for c, l in context2label.items()}
	pairs = [(l_s, l_c) for l_c in label2context for l_s in label2subword]
	legend_keys = [l_s + l_c for l_s, l_c in pairs]
	x = list(range(layers + 1))
	fig, (rg, ws, sl, sv) = plt.subplots(1, 4)
	keys = {'RG65' : rg, 'WS353' : ws, 'SL999' : sl, 'SV3500' : sv}
	colors = ['#a6611a', '#dfc27d', '#80cdc1', '#018571']
	linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
	line_properties = [(c, ls) for ls in linestyles for c in colors]
	for task in cont_scores: 
		for ((l_s, l_c), (color, linestyle)) in zip(pairs, line_properties):
			y = []
			s, c = label2subword[l_s], label2context[l_c]
			if c == 'decont':
				for i in range(layers + 1):
					y.append(decont_scores[task][(i, s)])
			else:
				y = cont_scores[task][(c, s)]
			keys[task].plot(x,y,linestyle=linestyle, color=color)
		keys[task].set_title("{}".format(task))
	# fig.suptitle(name2long[model_name])
	fig.set_size_inches(30, 10)
	plt.legend(legend_keys)
	# plt.show()
	plt.savefig('Figures/{}-repr_quality.jpeg'.format(model_name), format='jpeg', bbox_inches='tight', pad_inches=0.25)


# def submission_visualize(model_name, cont_scores, decont_scores, layers, decont_embedding_keys, cont_embedding_keys):
# 	subword2label = {"vec_min" : "f = min; ", "vec_max" : "f = max; ", "vec_last" : "f = last; ", "vec_mean" : "f = mean; "}
# 	context2label = {"min" : "g = min", "max" : "g = max", "mean" : "g = mean", "decont" : "g = decont"}
# 	label2subword, label2context = {l : s for s, l in subword2label.items()}, {l : c for c, l in context2label.items()}
# 	pairs = [(l_s, l_c) for l_c in label2context for l_s in label2subword]
# 	legend_keys = [l_s + l_c for l_s, l_c in pairs]
# 	x = list(range(layers + 1))
# 	fig, (rg, ws, sl, sv) = plt.subplots(1, 4)
# 	keys = {'RG65' : rg, 'WS353' : ws, 'SL999' : sl, 'SV3500' : sv}
# 	for task in cont_scores: 
# 		for l_s, l_c in pairs:
# 			y = []
# 			s, c = label2subword[l_s], label2context[l_c]
# 			if c == 'decont':
# 				for i in range(layers + 1):
# 					y.append(decont_scores[task][(i, s)])
# 			else:
# 				y = cont_scores[task][(c, s)]
# 			keys[task].plot(x,y)
# 		keys[task].set_title("{}".format(task))
# 	fig.suptitle(name2long[model_name])
# 	plt.legend(legend_keys)
# 	plt.show()


def submission_repr_quality_figures():
	RG65, WS353, SL999, SV3500, vocab, word_counts = data.fetch_wordsim_datasets(path_prefix)
	for model_name in tqdm(['gpt2sm', 'bertsm', 'robertasm', 'xlnetsm', 'distilbertsm', 'bertlg',  'gpt2lg', 'robertalg', 'xlnetlg']):
		n = 100000
		dataset = [None] * n
		# dataset = data.fetch_pooling_dataset(path_prefix, 'wikipedia_utf8_filtered_20pageviews.csv', 1000000, pickled=True, filter_vocab=vocab)
		# dataset = dataset[:n]
		if 'distil' in model_name:
			layers = 6
		elif 'sm' in model_name:
			layers = 12
		else:
			layers = 24
		decont_embeddings, decont_embedding_keys = distill.compute_decontextual(model_name, vocab, path_prefix='', pickled=True)
		decont_scores = scorer.score_decontextualized(decont_embeddings, layers, RG65, WS353, SL999, SV3500, decont_embedding_keys)

		cont_embeddings, cont_embedding_keys = distill.compute_contextual(dataset, model_name, vocab, path_prefix='', uniform='', pickled=True)
		cont_embeddings = fix_backups(cont_embeddings, decont_embeddings, vocab, cont_embedding_keys, layers)
		cont_scores, bests = scorer.score_contextualized(cont_embeddings, layers, RG65, WS353, SL999, SV3500, cont_embedding_keys)
		submission_visualize(model_name, cont_scores, decont_scores, layers, decont_embedding_keys, cont_embedding_keys)



def submission_repr_quality_tables():
	RG65, WS353, SL999, SV3500, vocab, word_counts = data.fetch_wordsim_datasets(path_prefix)
	initial_dataset = data.fetch_pooling_dataset(path_prefix, 'wikipedia_utf8_filtered_20pageviews.csv', 1000000, pickled=True, filter_vocab=vocab)
	for model_name in tqdm(['xlnetlg']):
		print('midrule')
		if 'distil' in model_name:
			layers = 6
		elif 'sm' in model_name:
			layers = 12
		else:
			layers = 24
		decont_embeddings, _ = distill.compute_decontextual(model_name, vocab, path_prefix='', pickled=True)
		for n in ([10000, 50000, 100000, 500000, 1000000]):
			dataset = initial_dataset[:n]
			cont_embeddings, cont_embedding_keys = distill.compute_contextual(dataset, model_name, vocab, path_prefix='', uniform='', pickled=True)

			layer_choice = layers // 4
			cont_embeddings = fix_backups(cont_embeddings, decont_embeddings, vocab, cont_embedding_keys, layers)
			cont_scores, bests = scorer.score_contextualized(cont_embeddings, layers, RG65, WS353, SL999, SV3500, cont_embedding_keys)

			RG65_l, RG65_s = bests['RG65']
			WS353_l, WS353_s = bests['WS353']
			SL999_l, SL999_s = bests['SL999']
			SV3500_l, SV3500_s = bests['SV3500']

			print('{} & {} & {} ({}) & {} ({}) & {} ({}) & {} ({})  \\\\'.format(name2long[model_name], n, RG65_s, RG65_l, WS353_s, WS353_l, SL999_s, SL999_l, SV3500_s, SV3500_l))
		print('bottomrule')
		exit()


def submission_bias_figures():
	class2words = garg_list.return_class2words()
	macro_pool_keys = ["min", "mean", "max"]
	micro_pool_keys = ["vec_min", "vec_max", "vec_mean", "vec_last"]
	cont_embedding_keys = [(a,b) for a in macro_pool_keys for b in micro_pool_keys]
	cartesian = [(a, n) for a in attributes.keys() for n in neutrals.keys()]
	macro_vocab = {w for (attribute, neutral) in cartesian for w in (attributes[attribute] | neutrals[neutral])}
	bolukbasi_color = '#dfc27d'
	garg_euc_color = '#80cdc1'
	garg_cos_color = '#018571'
	manzini_color = '#a6611a'
	adj_style = 'solid'
	prof_style = 'dashed'
	p_adj_style = 'dotted'
	p_prof_style = 'dashdot'
	for m_name in tqdm(['gpt2sm', 'bertsm', 'robertasm', 'distilbertsm', 'bertlg',  'gpt2lg', 'robertalg']):
		if 'distil' in m_name:
			layers = 6
		elif 'sm' in m_name:
			layers = 12
		else:
			layers = 24
		x = list(range(layers + 1))
		fig, (gen, rac, reli) = plt.subplots(1, 3)
		
		# GENDER PAIR
		attribute = 'gender_pair'
		neutral = 'adjectives'
		attribute_vocab = attributes[attribute]
		neutral_vocab = neutrals[neutral]
		N_attribute, V_attribute = dataset_lens[attribute], len(attribute_vocab)
		N_neutral, V_neutral = dataset_lens[neutral], len(neutral_vocab)

		model_name = m_name + attribute 
		attribute_embeddings =  pickle.load(open(model_name + '.{}_sents.vocab_size={}.contextualized.pickle'.format(N_attribute,V_attribute), 'rb'))
		decont_embeddings, _ = distill.compute_decontextual(m_name, attribute_vocab, path_prefix='', pickled=True)
		attribute_embeddings = fix_backups(attribute_embeddings, decont_embeddings, attribute_vocab, cont_embedding_keys, layers)

		model_name = m_name + neutral
		neutral_embeddings =  pickle.load(open(model_name + '.{}_sents.vocab_size={}.contextualized.pickle'.format(N_neutral,V_neutral), 'rb')) 
		decont_embeddings, _ = distill.compute_decontextual(m_name, neutral_vocab, path_prefix='', pickled=True)
		neutral_embeddings = fix_backups(neutral_embeddings, decont_embeddings, neutral_vocab, cont_embedding_keys, layers)

		bias_bolukbasi_list, bias_garg_euc_list, bias_garg_cos_list, bias_manzini_list = [], [], [], []
		for layer in range(layers + 1):
			l_attribute_embeddings = {w : attribute_embeddings[w][layer]['mean']['vec_mean'].numpy() for w in attribute_embeddings}
			l_neutral_embeddings = {w : neutral_embeddings[w][layer]['mean']['vec_mean'].numpy() for w in neutral_embeddings}
			bias_bolukbasi, bias_garg_euc, bias_garg_cos, bias_manzini = bias.compute_all_biases(l_attribute_embeddings, l_neutral_embeddings, attribute_vocab, neutral_vocab, gender_pair=True, classes=2, A1=None, A2=None, A_list=None)
			bias_bolukbasi_list.append(bias_bolukbasi)
			bias_garg_euc_list.append(bias_garg_euc)
			bias_garg_cos_list.append(bias_garg_cos)
			bias_manzini_list.append(bias_manzini)
		gen.plot(x, bias_bolukbasi_list, color = bolukbasi_color, linestyle = p_adj_style, label='P ADJ Bolukbasi')
		gen.plot(x, bias_garg_euc_list, color = garg_euc_color, linestyle = p_adj_style, label='P ADJ Garg-Euc')
		gen.plot(x, bias_garg_cos_list, color = garg_cos_color, linestyle = p_adj_style, label='P ADJ Garg-Cos')
		gen.plot(x, bias_manzini_list, color = manzini_color, linestyle = p_adj_style, label='P ADJ Manzini')

		neutral = 'professions'
		attribute_vocab = attributes[attribute]
		neutral_vocab = neutrals[neutral]
		N_attribute, V_attribute = dataset_lens[attribute], len(attribute_vocab)
		N_neutral, V_neutral = dataset_lens[neutral], len(neutral_vocab)

		model_name = m_name + attribute 
		attribute_embeddings =  pickle.load(open(model_name + '.{}_sents.vocab_size={}.contextualized.pickle'.format(N_attribute,V_attribute), 'rb'))
		decont_embeddings, _ = distill.compute_decontextual(m_name, attribute_vocab, path_prefix='', pickled=True)
		attribute_embeddings = fix_backups(attribute_embeddings, decont_embeddings, attribute_vocab, cont_embedding_keys, layers)

		model_name = m_name + neutral
		neutral_embeddings =  pickle.load(open(model_name + '.{}_sents.vocab_size={}.contextualized.pickle'.format(N_neutral,V_neutral), 'rb')) 
		decont_embeddings, _ = distill.compute_decontextual(m_name, neutral_vocab, path_prefix='', pickled=True)
		neutral_embeddings = fix_backups(neutral_embeddings, decont_embeddings, neutral_vocab, cont_embedding_keys, layers)

		bias_bolukbasi_list, bias_garg_euc_list, bias_garg_cos_list, bias_manzini_list = [], [], [], []
		for layer in range(layers + 1):
			l_attribute_embeddings = {w : attribute_embeddings[w][layer]['mean']['vec_mean'].numpy() for w in attribute_embeddings}
			l_neutral_embeddings = {w : neutral_embeddings[w][layer]['mean']['vec_mean'].numpy() for w in neutral_embeddings}
			bias_bolukbasi, bias_garg_euc, bias_garg_cos, bias_manzini = bias.compute_all_biases(l_attribute_embeddings, l_neutral_embeddings, attribute_vocab, neutral_vocab, gender_pair=True, classes=2, A1=None, A2=None, A_list=None)
			bias_bolukbasi_list.append(bias_bolukbasi)
			bias_garg_euc_list.append(bias_garg_euc)
			bias_garg_cos_list.append(bias_garg_cos)
			bias_manzini_list.append(bias_manzini)
		gen.plot(x, bias_bolukbasi_list, color = bolukbasi_color, linestyle = p_prof_style, label='P PROF Bolukbasi')
		gen.plot(x, bias_garg_euc_list, color = garg_euc_color, linestyle = p_prof_style, label='P PROF Garg-Euc')
		gen.plot(x, bias_garg_cos_list, color = garg_cos_color, linestyle = p_prof_style, label='P PROF Garg-Cos')
		gen.plot(x, bias_manzini_list, color = manzini_color, linestyle = p_prof_style, label='P PROF Manzini')

		# GENDER
		attribute = 'gender'
		A_list = [class2words[c] for c in {'male', 'female'}]
		A1, A2 = A_list
		neutral = 'adjectives'
		attribute_vocab = attributes[attribute]
		neutral_vocab = neutrals[neutral]
		N_attribute, V_attribute = dataset_lens[attribute], len(attribute_vocab)
		N_neutral, V_neutral = dataset_lens[neutral], len(neutral_vocab)

		model_name = m_name + attribute 
		attribute_embeddings =  pickle.load(open(model_name +  '.{}_sents.vocab_size={}.contextualized.pickle'.format(N_attribute,V_attribute), 'rb'))
		decont_embeddings, _ = distill.compute_decontextual(m_name, attribute_vocab, path_prefix='', pickled=True)
		attribute_embeddings = fix_backups(attribute_embeddings, decont_embeddings, attribute_vocab, cont_embedding_keys, layers)

		model_name = m_name + neutral
		neutral_embeddings =  pickle.load(open(model_name + '.{}_sents.vocab_size={}.contextualized.pickle'.format(N_neutral,V_neutral), 'rb')) 
		decont_embeddings, _ = distill.compute_decontextual(m_name, neutral_vocab, path_prefix='', pickled=True)
		neutral_embeddings = fix_backups(neutral_embeddings, decont_embeddings, neutral_vocab, cont_embedding_keys, layers)

		bias_bolukbasi_list, bias_garg_euc_list, bias_garg_cos_list, bias_manzini_list = [], [], [], []
		for layer in range(layers + 1):
			l_attribute_embeddings = {w : attribute_embeddings[w][layer]['mean']['vec_mean'].numpy() for w in attribute_embeddings}
			l_neutral_embeddings = {w : neutral_embeddings[w][layer]['mean']['vec_mean'].numpy() for w in neutral_embeddings}
			bias_bolukbasi, bias_garg_euc, bias_garg_cos, bias_manzini = bias.compute_all_biases(l_attribute_embeddings, l_neutral_embeddings, attribute_vocab, neutral_vocab, gender_pair=False, classes=2, A1=A1, A2=A2, A_list=A_list)
			bias_garg_euc_list.append(bias_garg_euc)
			bias_garg_cos_list.append(bias_garg_cos)
			bias_manzini_list.append(bias_manzini)
		gen.plot(x, bias_garg_euc_list, color = garg_euc_color, linestyle = adj_style, label='ADJ Garg-Euc')
		gen.plot(x, bias_garg_cos_list, color = garg_cos_color, linestyle = adj_style, label='ADJ Garg-Cos')
		gen.plot(x, bias_manzini_list, color = manzini_color, linestyle = adj_style, label='ADJ Manzini')

		neutral = 'professions'
		attribute_vocab = attributes[attribute]
		neutral_vocab = neutrals[neutral]
		N_attribute, V_attribute = dataset_lens[attribute], len(attribute_vocab)
		N_neutral, V_neutral = dataset_lens[neutral], len(neutral_vocab)

		model_name = m_name + attribute 
		attribute_embeddings =  pickle.load(open(model_name + '.{}_sents.vocab_size={}.contextualized.pickle'.format(N_attribute,V_attribute), 'rb'))
		decont_embeddings, _ = distill.compute_decontextual(m_name, attribute_vocab, path_prefix='', pickled=True)
		attribute_embeddings = fix_backups(attribute_embeddings, decont_embeddings, attribute_vocab, cont_embedding_keys, layers)

		model_name = m_name + neutral
		neutral_embeddings =  pickle.load(open(model_name + '.{}_sents.vocab_size={}.contextualized.pickle'.format(N_neutral,V_neutral), 'rb')) 
		decont_embeddings, _ = distill.compute_decontextual(m_name, neutral_vocab, path_prefix='', pickled=True)
		neutral_embeddings = fix_backups(neutral_embeddings, decont_embeddings, neutral_vocab, cont_embedding_keys, layers)

		bias_bolukbasi_list, bias_garg_euc_list, bias_garg_cos_list, bias_manzini_list = [], [], [], []
		for layer in range(layers + 1):
			l_attribute_embeddings = {w : attribute_embeddings[w][layer]['mean']['vec_mean'].numpy() for w in attribute_embeddings}
			l_neutral_embeddings = {w : neutral_embeddings[w][layer]['mean']['vec_mean'].numpy() for w in neutral_embeddings}
			bias_bolukbasi, bias_garg_euc, bias_garg_cos, bias_manzini = bias.compute_all_biases(l_attribute_embeddings, l_neutral_embeddings, attribute_vocab, neutral_vocab, gender_pair=False, classes=2, A1=A1, A2=A2, A_list=A_list)
			bias_garg_euc_list.append(bias_garg_euc)
			bias_garg_cos_list.append(bias_garg_cos)
			bias_manzini_list.append(bias_manzini)
		gen.plot(x, bias_garg_euc_list, color = garg_euc_color, linestyle = prof_style, label='PROF Garg-Euc')
		gen.plot(x, bias_garg_cos_list, color = garg_cos_color, linestyle = prof_style, label='PROF Garg-Cos')
		gen.plot(x, bias_manzini_list, color = manzini_color, linestyle = prof_style, label='PROF Manzini')

		# RACE
		A_list = [class2words[c] for c in {'white', 'hispanic', 'asian'}]
		attribute = 'race'
		neutral = 'adjectives'
		attribute_vocab = attributes[attribute]
		neutral_vocab = neutrals[neutral]
		N_attribute, V_attribute = dataset_lens[attribute], len(attribute_vocab)
		N_neutral, V_neutral = dataset_lens[neutral], len(neutral_vocab)

		model_name = m_name + attribute 
		attribute_embeddings =  pickle.load(open(model_name + '.{}_sents.vocab_size={}.contextualized.pickle'.format(N_attribute,V_attribute), 'rb'))
		decont_embeddings, _ = distill.compute_decontextual(m_name, attribute_vocab, path_prefix='', pickled=True)
		attribute_embeddings = fix_backups(attribute_embeddings, decont_embeddings, attribute_vocab, cont_embedding_keys, layers)

		model_name = m_name + neutral
		neutral_embeddings =  pickle.load(open(model_name + '.{}_sents.vocab_size={}.contextualized.pickle'.format(N_neutral,V_neutral), 'rb')) 
		decont_embeddings, _ = distill.compute_decontextual(m_name, neutral_vocab, path_prefix='', pickled=True)
		neutral_embeddings = fix_backups(neutral_embeddings, decont_embeddings, neutral_vocab, cont_embedding_keys, layers)

		bias_bolukbasi_list, bias_garg_euc_list, bias_garg_cos_list, bias_manzini_list = [], [], [], []
		for layer in range(layers + 1):
			l_attribute_embeddings = {w : attribute_embeddings[w][layer]['mean']['vec_mean'].numpy() for w in attribute_embeddings}
			l_neutral_embeddings = {w : neutral_embeddings[w][layer]['mean']['vec_mean'].numpy() for w in neutral_embeddings}
			bias_bolukbasi, bias_garg_euc, bias_garg_cos, bias_manzini = bias.compute_all_biases(l_attribute_embeddings, l_neutral_embeddings, attribute_vocab, neutral_vocab, gender_pair=False, classes=3, A1=None, A2=None, A_list=A_list)
			bias_manzini_list.append(bias_manzini)
		rac.plot(x, bias_manzini_list, color = manzini_color, linestyle = adj_style, label='ADJ Manzini')

		neutral = 'professions'
		attribute_vocab = attributes[attribute]
		neutral_vocab = neutrals[neutral]
		N_attribute, V_attribute = dataset_lens[attribute], len(attribute_vocab)
		N_neutral, V_neutral = dataset_lens[neutral], len(neutral_vocab)

		model_name = m_name + attribute 
		attribute_embeddings =  pickle.load(open(model_name + '.{}_sents.vocab_size={}.contextualized.pickle'.format(N_attribute,V_attribute), 'rb'))
		decont_embeddings, _ = distill.compute_decontextual(m_name, attribute_vocab, path_prefix='', pickled=True)
		attribute_embeddings = fix_backups(attribute_embeddings, decont_embeddings, attribute_vocab, cont_embedding_keys, layers)

		model_name = m_name + neutral
		neutral_embeddings =  pickle.load(open(model_name + '.{}_sents.vocab_size={}.contextualized.pickle'.format(N_neutral,V_neutral), 'rb')) 
		decont_embeddings, _ = distill.compute_decontextual(m_name, neutral_vocab, path_prefix='', pickled=True)
		neutral_embeddings = fix_backups(neutral_embeddings, decont_embeddings, neutral_vocab, cont_embedding_keys, layers)

		bias_bolukbasi_list, bias_garg_euc_list, bias_garg_cos_list, bias_manzini_list = [], [], [], []
		for layer in range(layers + 1):
			l_attribute_embeddings = {w : attribute_embeddings[w][layer]['mean']['vec_mean'].numpy() for w in attribute_embeddings}
			l_neutral_embeddings = {w : neutral_embeddings[w][layer]['mean']['vec_mean'].numpy() for w in neutral_embeddings}
			bias_bolukbasi, bias_garg_euc, bias_garg_cos, bias_manzini = bias.compute_all_biases(l_attribute_embeddings, l_neutral_embeddings, attribute_vocab, neutral_vocab, gender_pair=False, classes=3, A1=None, A2=None, A_list=A_list)
			bias_manzini_list.append(bias_manzini)
		rac.plot(x, bias_manzini_list, color = manzini_color, linestyle = prof_style, label='PROF Manzini')

		# RELIGION
		A_list = [class2words[c] for c in {'islam', 'christian'}]
		A1, A2 = A_list
		attribute = 'religion'
		neutral = 'adjectives'
		attribute_vocab = attributes[attribute]
		neutral_vocab = neutrals[neutral]
		N_attribute, V_attribute = dataset_lens[attribute], len(attribute_vocab)
		N_neutral, V_neutral = dataset_lens[neutral], len(neutral_vocab)

		model_name = m_name + attribute 
		attribute_embeddings =  pickle.load(open(model_name +  '.{}_sents.vocab_size={}.contextualized.pickle'.format(N_attribute,V_attribute), 'rb'))
		decont_embeddings, _ = distill.compute_decontextual(m_name, attribute_vocab, path_prefix='', pickled=True)
		attribute_embeddings = fix_backups(attribute_embeddings, decont_embeddings, attribute_vocab, cont_embedding_keys, layers)

		model_name = m_name + neutral
		neutral_embeddings =  pickle.load(open(model_name + '.{}_sents.vocab_size={}.contextualized.pickle'.format(N_neutral,V_neutral), 'rb')) 
		decont_embeddings, _ = distill.compute_decontextual(m_name, neutral_vocab, path_prefix='', pickled=True)
		neutral_embeddings = fix_backups(neutral_embeddings, decont_embeddings, neutral_vocab, cont_embedding_keys, layers)

		bias_bolukbasi_list, bias_garg_euc_list, bias_garg_cos_list, bias_manzini_list = [], [], [], []
		for layer in range(layers + 1):
			l_attribute_embeddings = {w : attribute_embeddings[w][layer]['mean']['vec_mean'].numpy() for w in attribute_embeddings}
			l_neutral_embeddings = {w : neutral_embeddings[w][layer]['mean']['vec_mean'].numpy() for w in neutral_embeddings}
			bias_bolukbasi, bias_garg_euc, bias_garg_cos, bias_manzini = bias.compute_all_biases(l_attribute_embeddings, l_neutral_embeddings, attribute_vocab, neutral_vocab, gender_pair=False, classes=2, A1=A1, A2=A2, A_list=A_list)
			bias_garg_euc_list.append(bias_garg_euc)
			bias_garg_cos_list.append(bias_garg_cos)
			bias_manzini_list.append(bias_manzini)
		reli.plot(x, bias_garg_euc_list, color = garg_euc_color, linestyle = adj_style, label='ADJ Garg-Euc')
		reli.plot(x, bias_garg_cos_list, color = garg_cos_color, linestyle = adj_style, label='ADJ Garg-Cos')
		reli.plot(x, bias_manzini_list, color = manzini_color, linestyle = adj_style, label='ADJ Manzini')

		neutral = 'professions'
		attribute_vocab = attributes[attribute]
		neutral_vocab = neutrals[neutral]
		N_attribute, V_attribute = dataset_lens[attribute], len(attribute_vocab)
		N_neutral, V_neutral = dataset_lens[neutral], len(neutral_vocab)

		model_name = m_name + attribute 
		attribute_embeddings =  pickle.load(open(model_name + '.{}_sents.vocab_size={}.contextualized.pickle'.format(N_attribute,V_attribute), 'rb'))
		decont_embeddings, _ = distill.compute_decontextual(m_name, attribute_vocab, path_prefix='', pickled=True)
		attribute_embeddings = fix_backups(attribute_embeddings, decont_embeddings, attribute_vocab, cont_embedding_keys, layers)

		model_name = m_name + neutral
		neutral_embeddings =  pickle.load(open(model_name + '.{}_sents.vocab_size={}.contextualized.pickle'.format(N_neutral,V_neutral), 'rb')) 
		decont_embeddings, _ = distill.compute_decontextual(m_name, neutral_vocab, path_prefix='', pickled=True)
		neutral_embeddings = fix_backups(neutral_embeddings, decont_embeddings, neutral_vocab, cont_embedding_keys, layers)

		bias_bolukbasi_list, bias_garg_euc_list, bias_garg_cos_list, bias_manzini_list = [], [], [], []
		for layer in range(layers + 1):
			l_attribute_embeddings = {w : attribute_embeddings[w][layer]['mean']['vec_mean'].numpy() for w in attribute_embeddings}
			l_neutral_embeddings = {w : neutral_embeddings[w][layer]['mean']['vec_mean'].numpy() for w in neutral_embeddings}
			bias_bolukbasi, bias_garg_euc, bias_garg_cos, bias_manzini = bias.compute_all_biases(l_attribute_embeddings, l_neutral_embeddings, attribute_vocab, neutral_vocab, gender_pair=False, classes=2, A1=A1, A2=A2, A_list=A_list)
			bias_garg_euc_list.append(bias_garg_euc)
			bias_garg_cos_list.append(bias_garg_cos)
			bias_manzini_list.append(bias_manzini)
		reli.plot(x, bias_garg_euc_list, color = garg_euc_color, linestyle = prof_style, label='PROF Garg-Euc')
		reli.plot(x, bias_garg_cos_list, color = garg_cos_color, linestyle = prof_style, label='PROF Garg-Cos')
		reli.plot(x, bias_manzini_list, color = manzini_color, linestyle = prof_style, label='PROF Manzini')
		
		gen.legend(shadow=True, fancybox=True)
		rac.legend(shadow=True, fancybox=True)
		reli.legend(shadow=True, fancybox=True)
		gen.set_title("{}".format('Gender'))
		rac.set_title("{}".format('Race'))
		reli.set_title("{}".format('Religion'))
		# fig.suptitle(name2long[m_name])
		print(m_name)
		fig.set_size_inches(30, 10)
		plt.savefig('Figures/{}-bias.jpeg'.format(m_name), format='jpeg', bbox_inches='tight', pad_inches=0.25)


def submission_bias_tables():
	class2words = garg_list.return_class2words()
	macro_pool_keys = ["min", "mean", "max"]
	micro_pool_keys = ["vec_min", "vec_max", "vec_mean", "vec_last"]
	cont_embedding_keys = [(a,b) for a in macro_pool_keys for b in micro_pool_keys]
	cartesian = [(a, n) for a in attributes.keys() for n in neutrals.keys()]
	macro_vocab = {w for (attribute, neutral) in cartesian for w in (attributes[attribute] | neutrals[neutral])}
	for model_name in (['word2vec', 'GloVe']): #bertsm', 'bertlg', 'gpt2sm', 'gpt2lg', 'robertasm', 'robertalg', 'xlnetsm', 'xlnetlg', 'distilbertsm']):
		bias_list = []
		if model_name == 'GloVe':
			embeddings, w2i, i2w = data.fetch_glove_embeddings(path_prefix, pickled=True)
		elif model_name == 'word2vec':
			embeddings = data.fetch_word2vec_embeddings(path_prefix, macro_vocab, pickled=True)
		else:
			if 'distil' in model_name:
				layers = 6
			elif 'sm' in model_name:
				layers = 12
			else:
				layers = 24
		for attribute in attributes:
			neutral = 'adjectives'
			attribute_vocab = attributes[attribute]
			neutral_vocab = neutrals[neutral]
			N_attribute, V_attribute = dataset_lens[attribute], len(attribute_vocab)
			N_neutral, V_neutral = dataset_lens[neutral], len(neutral_vocab)	
			
			if model_name == 'GloVe':
				attribute_embeddings, neutral_embeddings = {w : embeddings[w2i[w]] for w in attribute_vocab}, {w : embeddings[w2i[w]] for w in neutral_vocab}
			elif model_name == 'word2vec':
				attribute_embeddings = {w : embeddings[w] for w in attribute_vocab}
				neutral_embeddings = {w : embeddings[w] for w in neutral_vocab}
			else:
				layer = layers // 4 # chosen layer in paper's tables
				if 'xlnet' in model_name:
					attribute_embeddings =  pickle.load(open(model_name + '.{}_sents.vocab_size={}.contextualized.pickle'.format(N_attribute,V_attribute), 'rb'))
				else:
					attribute_embeddings =  pickle.load(open(model_name + attribute + '.{}_sents.vocab_size={}.contextualized.pickle'.format(N_attribute,V_attribute), 'rb'))
				decont_embeddings, _ = distill.compute_decontextual(model_name, attribute_vocab, path_prefix='', pickled=True)
				attribute_embeddings = fix_backups(attribute_embeddings, decont_embeddings, attribute_vocab, cont_embedding_keys, layers)
				attribute_embeddings = {w : attribute_embeddings[w][layer]['mean']['vec_mean'].numpy() for w in attribute_embeddings} # analyze f = mean, g = mean


				if 'xlnet' in model_name:
					neutral_embeddings=  pickle.load(open(model_name + '.{}_sents.vocab_size={}.contextualized.pickle'.format(N_neutral, V_neutral), 'rb'))
				else:
					neutral_embeddings =  pickle.load(open(model_name + neutral + '.{}_sents.vocab_size={}.contextualized.pickle'.format(N_neutral, V_neutral), 'rb'))
				decont_embeddings, _ = distill.compute_decontextual(model_name, neutral_vocab, path_prefix='', pickled=True)
				neutral_embeddings = fix_backups(neutral_embeddings, decont_embeddings, neutral_vocab, cont_embedding_keys, layers)
				neutral_embeddings = {w : neutral_embeddings[w][layer]['mean']['vec_mean'].numpy() for w in neutral_embeddings} # analyze f = mean, g = mean
			
			if attribute == 'gender_pair':
				bias_bolukbasi, bias_garg_euc, bias_garg_cos, bias_manzini = bias.compute_all_biases(attribute_embeddings, neutral_embeddings, attribute_vocab, neutral_vocab, gender_pair=True, classes=2, A1=None, A2=None, A_list=None)
				# biases[(bias2short[attribute], bias2short[neutral])].append('boluk: ' + str(bias_bolukbasi))
				# biases[(bias2short[attribute], bias2short[neutral])].append('g_euc: ' + str(bias_garg_euc))
				# biases[(bias2short[attribute], bias2short[neutral])].append('g_cos: ' + str(bias_garg_cos))
				bias_list.extend([bias_bolukbasi, bias_garg_euc, bias_garg_cos, bias_manzini])
			elif attribute == 'race':
				A_list = [class2words[c] for c in {'white', 'hispanic', 'asian'}]
				bias_bolukbasi, bias_garg_euc, bias_garg_cos, bias_manzini = bias.compute_all_biases(attribute_embeddings, neutral_embeddings, attribute_vocab, neutral_vocab, gender_pair=False, classes=3, A1=None, A2=None, A_list=A_list)
				bias_list.extend([bias_manzini])
			else:
				if attribute == 'gender':
					A_list = [class2words[c] for c in {'male', 'female'}]
					A1, A2 = A_list
			
				elif attribute == 'religion':
					A_list = [class2words[c] for c in {'islam', 'christian'}]
					A1, A2 = A_list
			
				else:
					raise NotImplementedError
				bias_bolukbasi, bias_garg_euc, bias_garg_cos, bias_manzini = bias.compute_all_biases(attribute_embeddings, neutral_embeddings, attribute_vocab, neutral_vocab, gender_pair=False, classes=2, A1=A1, A2=A2, A_list=A_list)
				bias_list.extend([bias_garg_euc, bias_garg_cos, bias_manzini])
		bias_list = [str(round(i, 4)) for i in bias_list]
		print(name2long[model_name] + ' & ' + ' & '.join(bias_list) +' \\\\')


def linearize_embeddings(pickle_f):
	print("Linearizing Embeddings for Analysis")
	embeddings = pickle.load(open(pickle_f, 'rb'))
	if 'glove' in pickle_f or 'static' in pickle_f:
		print("Linearized GloVe Embeddings")
		embedding_keys = set(embeddings[list(embeddings.keys())[0]].keys())
		return {k : {w : embeddings[w][k] for w in embeddings} for k in embedding_keys}
	elif '.contextualized' in pickle_f:
		print("Linearized Aggregated Embeddings")
		triples = [(l, macro, micro) for l in range(layers) for macro in macro_pool_keys for micro in micro_pool_keys]
		return {(l, macro, micro) : {w : embeddings[w][l][macro][micro].numpy() for w in embeddings} for (l, macro, micro) in triples}
	elif 'decontextualized' in pickle_f:
		print("Linearized Decontextualized Embeddings")
		embedding_keys = set(embeddings[list(embeddings.keys())[0]].keys())
		return {k : {w : embeddings[w][k].numpy() for w in embeddings} for k in embedding_keys}
	else:
		raise NotImplementedError


def main():
	submission_repr_quality_figures()
	submission_bias_figures()


if __name__ == '__main__':
	main()