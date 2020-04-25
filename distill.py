import torch
from transformers import *
import pickle
from tqdm import tqdm
from tokenization_bert import BertTokenizer, BasicTokenizer
from tokenization_gpt2 import GPT2Tokenizer
from tokenization_xlm import XLMTokenizer
from tokenization_xlnet import XLNetTokenizer
from tokenization_roberta import RobertaTokenizer
from tokenization_distilbert import DistilBertTokenizer

MODELS = {'bertsm'       : (BertModel,       BertConfig,       BertTokenizer,       'bert-base-uncased'),
		  'gpt2sm'       : (GPT2Model,       GPT2Config,       GPT2Tokenizer,       'gpt2'),
		  'xlnetsm'      : (XLNetModel,      XLNetConfig,      XLNetTokenizer,      'xlnet-base-cased'),
		  'robertasm'    : (RobertaModel,    RobertaConfig,    RobertaTokenizer,    'roberta-base'),
		  'distilbertsm' : (DistilBertModel, DistilBertConfig, DistilBertTokenizer, 'distilbert-base-uncased'),

		  'bertlg'    : (BertModel,    BertConfig,    BertTokenizer,      'bert-large-uncased'),
		  'gpt2lg'    : (GPT2Model,    GPT2Config,    GPT2Tokenizer,      'gpt2-medium'),
		  'xlnetlg'   : (XLNetModel,   XLNetConfig,   XLNetTokenizer,     'xlnet-large-cased'),
          'robertalg' : (RobertaModel, RobertaConfig, RobertaTokenizer,   'roberta-large'),


		  'xlm'     : (XLMModel,  XLMConfig,  XLMTokenizer,  'xlm-mlm-en-2048'),
		  'gpt2-36' : (GPT2Model, GPT2Config, GPT2Tokenizer, 'gpt2-large')}

name2layers = {'bertsm' : 12, 'gpt2sm' : 12, 'xlnetsm' : 12, 'robertasm' : 12, 'distilbertsm' : 6, 'bertlg':  24, 'gpt2lg' : 24, 'xlnetlg' : 24, 'robertalg' : 24, 'xlm' :  12, 'gpt2-36': 36}
macro_pool_keys = ["min", "mean", "max"]
micro_pool_keys = ["vec_min", "vec_max", "vec_mean", "vec_last"]


def compute_contextual(data, model_name, vocab, path_prefix='', uniform='', pickled=False):
	return compute_contextual_general(data, model_name, vocab, path_prefix=path_prefix, uniform=uniform, pickled=pickled)


def compute_decontextual(model_name, vocab, path_prefix='', pickled=False):
	return compute_decontextual_general(model_name, vocab, path_prefix=path_prefix, pickled=pickled)


def compute_contextual_general(data, model_name, vocab, path_prefix='', uniform='', pickled=False):
	word_occurences = {w: 0 for w in vocab}
	N, V = len(data), len(vocab)
	# print("Pooling over:", N)
	# print("Vocab Size:", V)
	# print("Uniform:", uniform)
	pickle_f = path_prefix + model_name + uniform + '.{}_sents.vocab_size={}.contextualized.pickle'.format(N,V)

	model_class, _, tokenizer_class, pretrained_weights = MODELS[model_name]
	tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
	model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
	model.eval()
	layers = name2layers[model_name]
	print_increments = set(range(0, N, N // 20))
	save_increments = {10000, 50000, 100000, 500000, 1000000}
	score_embeddings = {w : {i : {} for i in range(layers + 1)} for w in vocab}
	scores = []
	if pickled:
		score_embeddings = pickle.load(open(pickle_f, 'rb'))
	else:
		for elem, s in tqdm(list(enumerate(data))): # list is to enable tqdm
			if elem in print_increments:
				print(elem)
			if 'roberta' in model_name:
				subwords, indices, words = tokenizer.encode(s, add_special_tokens=True)
			else:
				subwords, indices, words = tokenizer.encode(s)
			assert len(indices) == len(words)
			input_ids = torch.tensor([subwords])
			encoded_layers = model(input_ids)[-1]
			for j, (index, w) in enumerate(zip(indices,words)):
				w = w.lower()
				start = index
				if j + 1 < len(indices):
					end = indices[j + 1]
				else:
					end = len(subwords)
				if w in vocab:
					word_occurences[w] += 1
					for i in range(layers + 1):
						vec = encoded_layers[i].squeeze(0)
						vec = vec.detach()
						vec_w = vec[start:end]
						vectors = {"vec_min": torch.min(vec_w, 0)[0], "vec_max" : torch.max(vec_w, 0)[0], "vec_last": vec_w[-1], "vec_mean": torch.mean(vec_w, 0)}
						if "min" in score_embeddings[w][i]:
							for key in score_embeddings[w][i]["min"]:
								score_embeddings[w][i]["min"][key] = torch.min(score_embeddings[w][i]["min"][key], vectors[key])
						else:
							score_embeddings[w][i]["min"] = {key : vectors[key] for key in vectors}
						if "mean" in score_embeddings[w][i]:
							for key in score_embeddings[w][i]["mean"]:
								old, new = word_occurences[w] - 1, word_occurences[w]
								score_embeddings[w][i]["mean"][key] = ((score_embeddings[w][i]["mean"][key] * old) + vectors[key]) / new
						else:
							score_embeddings[w][i]["mean"] = {key : vectors[key] for key in vectors}
						
						if "max" in score_embeddings[w][i]:
							for key in score_embeddings[w][i]["max"]:
								score_embeddings[w][i]["max"][key] = torch.max(score_embeddings[w][i]["max"][key], vectors[key])
						else:
							score_embeddings[w][i]["max"] = {key : vectors[key] for key in vectors}
			assert end == len(subwords)
			if elem in save_increments:
				print("Pickling at:", elem)
				pickle.dump(score_embeddings, open((path_prefix + model_name + uniform + '.{}_sents.vocab_size={}.contextualized.pickle'.format(elem,V)), 'wb'))		
		print("Final Pickling")
		pickle.dump(score_embeddings, open(pickle_f, 'wb'))			
	embedding_keys = [(a,b) for a in macro_pool_keys for b in micro_pool_keys]
	# print("Computed aggregated {} over {} sentences with {} words with uniform={}".format(model_name, N, V, uniform))
	return score_embeddings, embedding_keys


def compute_decontextual_general(model_name, vocab, path_prefix='', pickled=False):
	score_embeddings = {}
	V = len(vocab)
	pickle_f = path_prefix + model_name + 'vocab_size={}.decontextualized.pickle'.format(V)
	model_class, _, tokenizer_class, pretrained_weights = MODELS[model_name]
	tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
	model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
	model.eval()
	layers = name2layers[model_name]
	
	embedding_keys = [(i, k) for k in micro_pool_keys for i in range(layers + 1)]
	if pickled:
		score_embeddings = pickle.load(open(pickle_f, 'rb'))
	else:
		for w in tqdm(vocab):
			score_embeddings[w] = {}
			if 'roberta' in model_name:
				s, indices, words = tokenizer.encode(' ' + w, add_special_tokens=True)
			else:
				s, indices, words = tokenizer.encode(w)
			if len(words) != 1:
				print(model_name, w)
			input_ids = torch.tensor([s])
			encoded_layers = model(input_ids)[-1]
			for i in range(0, layers + 1):
				vec_w = encoded_layers[i]
				vec_w = vec_w.detach()
				vec_w = vec_w.squeeze(0)
				if 'roberta' in model_name:
					vec_w = vec_w[1:-1]
				vec_last = vec_w[-1]
				vec_max = torch.max(vec_w, 0)[0]
				vec_mean = torch.mean(vec_w, 0)
				vec_min = torch.min(vec_w, 0)[0]
				vec_w_micro = {"vec_min" : vec_min, "vec_max" : vec_max, "vec_mean" : vec_mean, "vec_last": vec_last}	
				for k in micro_pool_keys:		
					score_embeddings[w][(i, k)] = vec_w_micro[k]
		print("Pickling decontextualized {}".format(model_name))
		pickle.dump(score_embeddings, open(pickle_f, 'wb'))
	# print("Computed decontextualized {}".format(model_name))
	return score_embeddings, embedding_keys
