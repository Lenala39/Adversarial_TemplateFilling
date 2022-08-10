import os

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
from numpy.testing import assert_allclose
from torch import embedding

class USE:
    def __init__(self):
        print("Loading USE from tf.hub")
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/4") # -large
    
    def semantic_similarity(self, sent1, sent2):
        embeddings = self.model([sent1, sent2])
        similarity = np.inner(embeddings["outputs"], embeddings["outputs"])
       
        #assert assert_allclose(similarity[0][1], similarity[1][0], rtol=1e-5, atol=0)
        #assert assert_allclose(similarity[0][0], 1.0, rtol=1e-5, atol=0)
        #assert assert_allclose(similarity[1][1], 1.0, rtol=1e-5, atol=0)
        
        return similarity[0][1]

    def semantic_similarty_list(self, sentences_new, sentences_original):
        embeddings_new = self.model(sentences_new)
        embeddings_original = self.model(sentences_original)
        similarity = np.inner(embeddings_new["outputs"], embeddings_original["outputs"])
        relevant_similarities = [similarity[i][i] for i in range(len(similarity))]
        
        return relevant_similarities


def similarity_calculation(indices, orig_texts, new_texts, sim_predictor, thres, attack_types=None):
    # compute semantic similarity
    half_sim_window = (thres['sim_window'] - 1) // 2
    orig_locals = []
    new_locals = []
    for i in range(len(indices)):
        idx = indices[i]
        len_text = len(orig_texts[i])
        if idx >= half_sim_window and len_text - idx - 1 >= half_sim_window:
            text_range_min = idx - half_sim_window
            text_range_max = idx + half_sim_window + 1
        elif idx < half_sim_window and len_text - idx - 1 >= half_sim_window:
            text_range_min = 0
            text_range_max = thres['sim_window']
        elif idx >= half_sim_window and len_text - idx - 1 < half_sim_window:
            text_range_min = len_text - thres['sim_window']
            text_range_max = len_text
        else:
            text_range_min = 0
            text_range_max = len_text
        orig_locals.append(" ".join(orig_texts[i][text_range_min:text_range_max]))
        
        if attack_types[i] == 'merge': 
            text_range_max -= 1
        if attack_types[i] == 'insert' and text_range_min > 0:
            text_range_min -= 1
        new_locals.append(" ".join(new_texts[i][max(text_range_min,0):text_range_max]))
    
    return sim_predictor.semantic_similarty_list(orig_locals, new_locals)