import re
import six
import collections
import numpy as np
from tqdm import tqdm
from konlpy.tag import Mecab
from rouge_score import scoring

def _create_ngrams(tokens, n):
	"""Creates ngrams from the given list of tokens.
	Args:
		tokens: A list of tokens from which ngrams are created.
		n: Number of tokens to use, e.g. 2 for bigrams.
	Returns:
		A dictionary mapping each bigram to the number of occurrences.
	"""

	ngrams = collections.Counter()
	for ngram in (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)):
		ngrams[ngram] += 1
	return ngrams


def _score_lcs(target_tokens, prediction_tokens):
	"""Computes LCS (Longest Common Subsequence) rouge scores.
	Args:
		target_tokens: Tokens from the target text.
		prediction_tokens: Tokens from the predicted text.
	Returns:
		A Score object containing computed scores.
	"""

	if not target_tokens or not prediction_tokens:
		return scoring.Score(precision=0, recall=0, fmeasure=0)

	# Compute length of LCS from the bottom up in a table (DP appproach).
	lcs_table = _lcs_table(target_tokens, prediction_tokens)
	lcs_length = lcs_table[-1][-1]

	precision = lcs_length / len(prediction_tokens)
	recall = lcs_length / len(target_tokens)
	fmeasure = scoring.fmeasure(precision, recall)

	return scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure)


def _lcs_table(ref, can):
	"""Create 2-d LCS score table."""
	rows = len(ref)
	cols = len(can)
	lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
	for i in range(1, rows + 1):
		for j in range(1, cols + 1):
			if ref[i - 1] == can[j - 1]:
				lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
			else:
				lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
	return lcs_table


def _backtrack_norec(t, ref, can):
	"""Read out LCS."""
	i = len(ref)
	j = len(can)
	lcs = []
	while i > 0 and j > 0:
		if ref[i - 1] == can[j - 1]:
			lcs.insert(0, i-1)
			i -= 1
			j -= 1
		elif t[i][j - 1] > t[i - 1][j]:
			j -= 1
		else:
			i -= 1
	return lcs


def _summary_level_lcs(ref_sent, can_sent):
	"""ROUGE: Summary-level LCS, section 3.2 in ROUGE paper.
	Args:
		ref_sent: list of tokenized reference sentences
		can_sent: list of tokenized candidate sentences
	Returns:
		summary level ROUGE score
	"""
	if not ref_sent or not can_sent:
		return scoring.Score(precision=0, recall=0, fmeasure=0)

	m = sum(map(len, ref_sent))
	n = sum(map(len, can_sent))
	if not n or not m:
		return scoring.Score(precision=0, recall=0, fmeasure=0)

	# get token counts to prevent double counting
	token_cnts_r = collections.Counter()
	token_cnts_c = collections.Counter()
	for s in ref_sent:
		# s is a list of tokens
		token_cnts_r.update(s)
	for s in can_sent:
		token_cnts_c.update(s)

	hits = 0
	for r in ref_sent:
		lcs = _union_lcs(r, can_sent)
		# Prevent double-counting:
		# The paper describes just computing hits += len(_union_lcs()),
		# but the implementation prevents double counting. We also
		# implement this as in version 1.5.5.
		for t in lcs:
			if token_cnts_c[t] > 0 and token_cnts_r[t] > 0:
				hits += 1
				token_cnts_c[t] -= 1
				token_cnts_r[t] -= 1

	recall = hits / m
	precision = hits / n
	fmeasure = scoring.fmeasure(precision, recall)
	return scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure)


def _union_lcs(ref, c_list):
	"""Find union LCS between a ref sentence and list of candidate sentences.
	Args:
		ref: list of tokens
		c_list: list of list of indices for LCS into reference summary
	Returns:
		List of tokens in ref representing union LCS.
	"""
	lcs_list = [lcs_ind(ref, c) for c in c_list]
	return [ref[i] for i in _find_union(lcs_list)]


def _find_union(lcs_list):
	"""Finds union LCS given a list of LCS."""
	return sorted(list(set().union(*lcs_list)))


def lcs_ind(ref, can):
	"""Returns one of the longest lcs."""
	t = _lcs_table(ref, can)
	return _backtrack_norec(t, ref, can)


def _score_ngrams(target_ngrams, prediction_ngrams):
	"""Compute n-gram based rouge scores.
	Args:
		target_ngrams: A Counter object mapping each ngram to number of
		occurrences for the target text.
		prediction_ngrams: A Counter object mapping each ngram to number of
		occurrences for the prediction text.
	Returns:
		A Score object containing computed scores.
	"""

	intersection_ngrams_count = 0
	for ngram in six.iterkeys(target_ngrams):
		intersection_ngrams_count += min(target_ngrams[ngram],prediction_ngrams[ngram])

	target_ngrams_count = sum(target_ngrams.values())
	prediction_ngrams_count = sum(prediction_ngrams.values())

	precision = intersection_ngrams_count / max(prediction_ngrams_count, 1)
	recall = intersection_ngrams_count / max(target_ngrams_count, 1)
	fmeasure = scoring.fmeasure(precision, recall)

	return scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure)


class KoreanRougeScorer(scoring.BaseScorer):
    def __init__(self, rouge_types):
        self.rouge_types = rouge_types
        self.tokenizer = Mecab()

    def score(self, target, prediction):
        if len(self.rouge_types) == 1 and self.rouge_types[0] == "rougeLsum":
            target_tokens = None
            prediction_tokens = None
        else:
            target_tokens = self.tokenizer.morphs(target)
            prediction_tokens = self.tokenizer.morphs(prediction) 
        result = {}

        for rouge_type in self.rouge_types:
            if rouge_type == "rougeL":
                scores = _score_lcs(target_tokens, prediction_tokens)
            elif re.match(r"rouge[0-9]$", six.ensure_str(rouge_type)):
                # Rouge from n-grams.
                n = int(rouge_type[5:])
                if n <= 0:
                    raise ValueError("rougen requires positive n: %s" % rouge_type)
            
                target_ngrams = _create_ngrams(target_tokens, n)
                prediction_ngrams = _create_ngrams(prediction_tokens, n)
                scores = _score_ngrams(target_ngrams, prediction_ngrams)
            else:
                raise ValueError("Invalid rouge type: %s" % rouge_type)
            result[rouge_type] = scores
        return result

def compute(predictions, references, use_agregator=True):
	rouge_types = ["rouge1", "rouge2", "rougeL"]
	scorer = KoreanRougeScorer(rouge_types=rouge_types)

	if use_agregator:
		aggregator = scoring.BootstrapAggregator()
	else:
		scores = []

	for ref, pred in zip(references, predictions):
		score = scorer.score(ref, pred)
		if use_agregator:
			aggregator.add_scores(score)
		else:
			scores.append(score)

	if use_agregator:
		result = aggregator.aggregate()
	else:
		result = {}
		for key in scores[0]:
			result[key] = list(score[key] for score in scores)
	return result


def compute_metrics(eval_pred, tokenizer):
	predictions, labels = eval_pred
	
	for i, pred in enumerate(predictions) :
		pred_vector = pred.tolist()
		if tokenizer.pad_token_id in pred_vector :
			index = pred_vector.index(tokenizer.pad_token_id)
			predictions[i][index+1:] = tokenizer.pad_token_id
			
	decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
	labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
	decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
	decoded_preds = [pred.strip() for pred in decoded_preds]
	decoded_labels = [label.strip() for label in decoded_labels]
	result = compute(predictions=decoded_preds, references=decoded_labels)
    
    # Extract a few results
	result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
	prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
	result["gen_len"] = np.mean(prediction_lens)
	return {k: round(v, 4) for k, v in result.items()}
    