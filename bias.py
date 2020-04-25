import numpy as np
from functools import reduce
from sklearn.decomposition import PCA
import sklearn
from scipy import spatial
import matplotlib.pyplot as plt
import garg_list


seed_pairs = {('she', 'he'), ('her', 'his'), ('woman', 'man'), ('mary', 'john'), ('herself', 'himself'), ('daughter', 'son'), ('mother', 'father'), ('gal', 'guy'), ('girl', 'boy'), ('female', 'male')}
class2words = garg_list.return_class2words()
adjectives = garg_list.return_adjectives()
professions = garg_list.return_professions()


def fetch_seed_pairs_vocab():
	return {x for x, _ in seed_pairs} | {x for _, x in seed_pairs}


def fetch_class_words(classes):
	return set.union(*[class2words[c] for c in classes])


def fetch_adjectives():
	return adjectives


def fetch_professions():
	return professions


def compute_bias_direction(embeddings):
	g_sum = None
	diff_embeddings = [embeddings[x] - embeddings[y] for x,y in seed_pairs]
	X = np.array(diff_embeddings)
	pca = PCA(n_components=1)
	pca.fit(X)
	return pca.components_[0], pca.explained_variance_ratio_[0]


def compute_garg_euc_bias(attribute_embeddings, neutral_embeddings, A1, A2):
	a1, a2 = sum([attribute_embeddings[w] for w in A1]) / len(A1), sum([attribute_embeddings[w] for w in A2]) / len(A2)
	return sum([abs(np.linalg.norm(neutral_embeddings[w] - a1) - np.linalg.norm(neutral_embeddings[w] - a2)) for w in neutral_embeddings]) / len(neutral_embeddings)


def compute_garg_cos_bias(attribute_embeddings, neutral_embeddings, A1, A2):
	a1, a2 = sum([attribute_embeddings[w] for w in A1]) / len(A1), sum([attribute_embeddings[w] for w in A2]) / len(A2)
	return sum([abs((1 - spatial.distance.cosine(neutral_embeddings[w], a1)) - (1 - spatial.distance.cosine(neutral_embeddings[w], a2))) for w in neutral_embeddings]) / len(neutral_embeddings)


def compute_manzini_bias(attribute_embeddings, neutral_embeddings, A_list):
	a_list = [(sum([attribute_embeddings[w] for w in Ai]) / len(Ai)) for Ai in A_list]
	neutral_bias = []
	for w, e in neutral_embeddings.items():
		neutral_bias.append(abs(sum([(1 - spatial.distance.cosine(e, ai)) for ai in a_list]) / len(a_list)))
	assert len(neutral_bias) == len(neutral_embeddings)
	return sum(neutral_bias) / len(neutral_bias)


	a1, a2 = sum([attribute_embeddings[w] for w in A1]) / len(A1), sum([attribute_embeddings[w] for w in A2]) / len(A2)
	return sum([abs((1 - spatial.distance.cosine(neutral_embeddings[w], a1)) - (1 - spatial.distance.cosine(neutral_embeddings[w], a2))) for w in neutral_embeddings]) / len(neutral_embeddings)

def compute_mean_difference(embeddings):
	diff_embeddings = [embeddings[x] - embeddings[y] for x,y in seed_pairs]
	return sum(diff_embeddings) / len(diff_embeddings)


def compute_mean(embeddings, words):
	return sum([embeddings[x] for x in words]) / len(words)


def compute_bias(embeddings, g, pool):
	return sum([pool(1 - spatial.distance.cosine(embeddings[w],g)) for w in embeddings]) / len(embeddings)


def compute_multiclass_bias(neutral_embeddings, class_embeddings, classes):
	K = len(classes)
	class2vec = {c : compute_mean(class_embeddings, class2words[c]) for c in classes}
	class2sim = {c : compute_bias(neutral_embeddings, class2vec[c], lambda x: max(0,x)) for c in classes}
	uniform = 1 / K
	Z = sum(class2sim.values()) 
	sim_dist = [v / Z for v in class2sim.values()] # Assumes class2sim.values() is non-negative 
	unscaled_bias = max([abs(s - uniform) for s in sim_dist])
	return unscaled_bias * K / (K - 1) 


def visualize_bias(linearized_gender_embeddings, linearized_profession_embeddings, model_type, layers, legend_keys, aggregated=False, static=True):
	key_list = sorted(linearized_gender_embeddings.keys())
	if static:
		y_pos = np.arange(len(key_list))
		performance = []
		for key in key_list:
			print("Analyzing Gender Bias of:", key)
			g, variance_explained = compute_bias_direction(linearized_gender_embeddings[key])
			bias_value = compute_bias(linearized_profession_embeddings[key], g, abs)
			print(bias_value)
			performance.append(bias_value)
		plt.bar(y_pos, performance, align='center', alpha=0.5)
		plt.xticks(y_pos, key_list)
		plt.title("Bias for Static Embeddings - Normal style graph")	
		plt.show()
	else:
		x = list(range(1, layers + 1))
		ys = {k : [] for k in legend_keys}
		if aggregated:
			micro = "vec_mean"
			for key in key_list:
				if key[2] == micro:
					g, variance_explained = compute_bias_direction(linearized_gender_embeddings[key])
					bias_value = compute_bias(linearized_profession_embeddings[key], g, abs)
					print("Gender bias of layer {} with macro pooling {} is {}".format(key[0], key[1], bias_value))
					ys[key[1]].append(bias_value)
			for key in legend_keys:
				plt.plot(x, ys[key])
			plt.title("Bias for Aggregated {} - Normal style graph".format(model_type))
			plt.legend(legend_keys)
			plt.show()
		else:
			for key in key_list:
				g, variance_explained = compute_bias_direction(linearized_gender_embeddings[key])
				bias_value = compute_bias(linearized_profession_embeddings[key], g, abs)
				print("Gender bias of layer {} with subword pooling {} is {}".format(key[0], key[1], bias_value))
				ys[key[1]].append(bias_value)
			for key in legend_keys:
				plt.plot(x, ys[key])
			plt.title("Bias for Decontextualized {} - Normal style graph".format(model_type))
			plt.legend(legend_keys)
			plt.show()


def visualize_garg_bias(linearized_gender_embeddings, linearized_profession_embeddings, model_type, layers, legend_keys, aggregated=False, static=True):
	key_list = sorted(linearized_gender_embeddings.keys())
	if static:
		y_pos = np.arange(len(key_list))
		performance = []
		for key in key_list:
			print("Analyzing Gender Bias of:", key)
			g = compute_mean_difference(linearized_gender_embeddings[key])
			bias_value = compute_bias(linearized_profession_embeddings[key], g, abs)
			print(bias_value)
			performance.append(bias_value)
		plt.bar(y_pos, performance, align='center', alpha=0.5)
		plt.xticks(y_pos, key_list)
		plt.title("Garg-Style Gender Bias for Static Embeddings")	
		plt.show()
	else:
		x = list(range(1, layers + 1))
		ys = {k : [] for k in legend_keys}
		if aggregated:
			micro = "vec_mean"
			for key in key_list:
				if key[2] == micro:
					g = compute_mean_difference(linearized_gender_embeddings[key])
					bias_value = compute_bias(linearized_profession_embeddings[key], g, abs)
					print("Gender bias of layer {} with macro pooling {} is {}".format(key[0], key[1], bias_value))
					ys[key[1]].append(bias_value)
			for key in legend_keys:
				plt.plot(x, ys[key])
			plt.title("Garg-Style Gender Bias for Aggregated {}".format(model_type))
			plt.legend(legend_keys)
			plt.show()
		else:
			for key in key_list:
				g = compute_mean_difference(linearized_gender_embeddings[key])
				bias_value = compute_bias(linearized_profession_embeddings[key], g, abs)
				print("Gender bias of layer {} with subword pooling {} is {}".format(key[0], key[1], bias_value))
				ys[key[1]].append(bias_value)
			for key in legend_keys:
				plt.plot(x, ys[key])
			plt.title("Garg-Style Gender Bias for Decontextualized {}".format(model_type))
			plt.legend(legend_keys)
			plt.show()


def visualize_multiclass_bias(class_embeddings, neutral_embeddings, classes, model_type, layers, legend_keys, aggregated=False, static=True):
	key_list = sorted(class_embeddings.keys())
	print("Computing Multiclass Bias")
	if static:
		y_pos = np.arange(len(key_list))
		performance = []
		for key in key_list:
			print("Analyzing Multiclass Bias of:", key)
			bias_value = compute_multiclass_bias(neutral_embeddings[key], class_embeddings[key], classes)
			print(bias_value)
			performance.append(bias_value)
		plt.bar(y_pos, performance, align='center', alpha=0.5)
		plt.xticks(y_pos, key_list)
		plt.title("Max-Style Multiclass Bias for {} Classes for Static Embeddings".format(classes))	
		plt.show()
	else:
		x = list(range(1, layers + 1))
		ys = {k : [] for k in legend_keys}
		if aggregated:
			micro = "vec_mean"
			for key in key_list:
				if key[2] == micro:
					print("Analyzing Multiclass Bias of:", key)
					bias_value = compute_multiclass_bias(neutral_embeddings[key], class_embeddings[key], classes)
					print(bias_value)
					print("Multiclass Bias of layer {} with macro pooling {} is {}".format(key[0], key[1], bias_value))
					ys[key[1]].append(bias_value)
			for key in legend_keys:
				plt.plot(x, ys[key])
			plt.title("Max-Style Multiclass Bias for Aggregated {}".format(model_type))
			plt.legend(legend_keys)
			plt.show()
		else:
			for key in key_list:
				print("Analyzing Multiclass Bias of:", key)
				bias_value = compute_multiclass_bias(neutral_embeddings[key], class_embeddings[key], classes)
				print(bias_value)
				print("Multiclass Bias of layer {} with subword pooling {} is {}".format(key[0], key[1], bias_value))
				ys[key[1]].append(bias_value)
			for key in legend_keys:
				plt.plot(x, ys[key])
			plt.title("Max-Style Multiclass Bias for Decontextualized {}".format(model_type))
			plt.legend(legend_keys)
			plt.show()


def compute_all_biases(attribute_embeddings, neutral_embeddings, attribute_vocab, neutral_vocab, gender_pair, classes, A1=None, A2=None, A_list=None):
	bias_bolukbasi, bias_garg_euc, bias_garg_cos, bias_manzini = None, None, None, None
	if gender_pair: # Set P exists
		g, pca_variance_explained = compute_bias_direction(attribute_embeddings)
		bias_bolukbasi = compute_bias(neutral_embeddings, g, abs)
		A1, A2 = [x for (x,y) in seed_pairs], [y for (x,y) in seed_pairs]
		A_list = [A1, A2]

	if classes == 2: # Binary bias
		bias_garg_euc = compute_garg_euc_bias(attribute_embeddings, neutral_embeddings, A1, A2)
		bias_garg_cos = compute_garg_cos_bias(attribute_embeddings, neutral_embeddings, A1, A2)
	
	bias_manzini = compute_manzini_bias(attribute_embeddings, neutral_embeddings, A_list)
	return bias_bolukbasi, bias_garg_euc, bias_garg_cos, bias_manzini