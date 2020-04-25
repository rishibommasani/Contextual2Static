import sklearn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from scipy.stats.mstats import spearmanr


def score_contextualized(embeddings, layers, RG, WS, SL, SV, embedding_keys):
	originals = {"RG65" : RG, "WS353" : WS, "SL999" : SL, "SV3500" : SV}
	scores = {"RG65" : {}, "WS353" : {}, "SL999" : {}, "SV3500" : {}}
	bests = {"RG65" : {}, "WS353" : {}, "SL999" : {}, "SV3500" : {}}
	for key in scores:
		for macro, micro in embedding_keys:
			scores[key][(macro, micro)] = []
			for i in range(layers + 1):
				scores_human = []
				scores_embed = []
				for w1, w2, score in originals[key]:
					e1 = embeddings[w1][i][macro][micro]
					e2 = embeddings[w2][i][macro][micro]
					cos = 1 - cosine(e1, e2)
					scores_human.append(score)
					scores_embed.append(cos)
				scores[key][(macro, micro)].append(spearmanr(scores_human, scores_embed)[0])	
	for key in scores:
		l = [(i,round(v,4)) for i, v in enumerate(scores[key][('mean', 'vec_mean')])]
		bests[key] = sorted(l, key = lambda t: t[1])[-1] 	
	return scores, bests


def score_decontextualized(embeddings, layers, RG, WS, SL, SV, embedding_keys):
	originals = {"RG65" : RG, "WS353" : WS, "SL999" : SL, "SV3500" : SV}
	scores = {"RG65" : {}, "WS353" : {}, "SL999" : {}, "SV3500" : {}}
	for key in scores:
		for embedding_key in embedding_keys:
			scores_human = []
			scores_embed = []
			for w1, w2, score in originals[key]:
				e1 = embeddings[w1][embedding_key]
				e2 = embeddings[w2][embedding_key]
				cos = 1 - cosine(e1, e2)
				scores_human.append(score)
				scores_embed.append(cos)
			scores[key][embedding_key]= round(spearmanr(scores_human, scores_embed)[0], 4)	
	return scores