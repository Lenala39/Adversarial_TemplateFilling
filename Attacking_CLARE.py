#import tensorflow as tf

from pydoc import doc
from deprecated import deprecated
import numpy as np
from sklearn import model_selection
from torch import sigmoid 

from transformers import RobertaTokenizer
from transformers import pipeline
from transformers import FillMaskPipeline
from transformers import RobertaForMaskedLM
from DocumentChunking import DocumentChunking
from DocumentEncoder import DocumentEncoder

from similarity_model import USE, similarity_calculation
from SlotFillingCompModule import SlotFillingCompModule, create_batches_from_chunking, prepare_module
import EntityAligner
import EntityDecoder
from helper_functions_CLARE import *
from helper_functions import * 
from helper_functions import __compute_divergence__ 

from official.nlp.bert import tokenization
import tensorflow_hub as hub
import tensorflow as tf 
from copy import deepcopy

import json 
import os 

from Chunk import Chunk

from collections import Counter 


bert_model_name = "https://tfhub.dev/google/experts/bert/pubmed/2"

import nltk

pos_tag_filter = {
                    'replace': set(['NOUN', 'VERB', 'ADJ', 'X', 'NUM', 'ADV']),
                    'insert': set(['NOUN/NOUN', 'ADJ/NOUN', 'NOUN/VERB', 'NOUN/ADP',
                               'ADP/NOUN', 'NOUN/.', 'VERB/NOUN', 'DET/NOUN',
                               'VERB/ADJ', './NOUN', 'VERB/VERB', 'VERB/DET',
                               'DET/ADJ', 'ADJ/ADJ', 'VERB/ADP', 'NOUN/CONJ',
                               'NOUN/ADJ', 'PRT/VERB', 'ADP/DET', 'ADP/ADJ',
                               'PRON/NOUN', 'VERB/PRON', './X', './DET']),
                    'merge': set(['NOUN/NOUN', 'ADJ/NOUN', 'VERB/ADJ', 'VERB/NOUN',
                              'VERB/VERB', 'NOUN/VERB', 'DET/ADJ', 'ADJ/ADJ',
                              'DET/NOUN', 'NUM/NOUN', 'PRON/NOUN', 'NOUN/ADJ',
                              'ADV/VERB', 'VERB/ADV', 'PRON/ADJ'])
                 }

thres = {'replace_prob': 0.0005, 'insert_prob': 0.0, 'merge_prob': 0.005,
            'replace_sim': 0.6, 'insert_sim': -1.0, 'merge_sim': 0.7,
            'prob_diff': -5e-4, 'sim_window': 15, 'keep_sim': True,
            'divergence_thres_extraction': 0.1, 'divergence_thres_comp':0.0}

binary_words = set(['is', 'are', 'was', 'were', "'s", "'re", "'ll", "will",
                    "could", "would", "may", "might", "can", "must",
                    'has', 'have', 'not', 'no', 'nor', "'t", 'wont'])

import re
punct_re = re.compile(r'\W')
words_re = re.compile(r'\w')

disease_str, victim_type, start, end, comp_mode, data_mode = parse_cli_arguments()

if disease_str == "gl":
    model_prefix = "glaucoma"
else: 
    model_prefix = "dm2"
dir_path = os.path.dirname(os.path.realpath(__file__)) 

# create BERT tokenizer
FullTokenizer = tokenization.FullTokenizer
bert_layer = hub.KerasLayer(bert_model_name, trainable=False)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
bert_tokenizer = FullTokenizer(vocab_file)

# construct the module for data and victim model
slot_filling_module = prepare_module(model_prefix=model_prefix, load_from_pickle=True, train=False, model="CLARE", 
                                        comp_mode=comp_mode, load_augmented=False, victim_type=victim_type,
                                        load_slot_indices=True, model_name=None, load_weights=True)

victim = slot_filling_module.slot_filling_model

print("Creating Document Encoder")
# create document encoder
document_encoder = DocumentEncoder(bert_tokenizer, slot_filling_module.slot_indices, 'no_slot')

print("Creating Roberta")
# prepare context predictor
roberta_tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
roberta = RobertaForMaskedLM.from_pretrained('distilroberta-base')
roberta_mlm = FillMaskPipeline(roberta, roberta_tokenizer) # can set top_k = int for determining how many synonyms to return (default 5)

use_sim = USE()

print("Loading stop words")
stop_words_set = set(nltk.corpus.stopwords.words('english'))

def apply_clare(document, doc_chunking, batch, chunk, sentence_obj, original_prediction, original_prediction_split, victim_type="comp"):
       
    sentence = sentence_obj.get_tokens()
    sentence_original = deepcopy(sentence)
    num_queries = 1
    
    # find attack sequences according to predicted probablity change
    attack_sequences, num_query = get_attack_sequences(sentence=sentence, sentence_obj=sentence_obj, chunk=chunk, doc_chunking=doc_chunking, 
                                                        original_prediction=original_prediction, original_prediction_split=original_prediction_split, document=document, attack_loc="pos_tag_filter")

    num_queries += num_query

    attack_logs = []
    sentence_copy = sentence.copy()

    insertions = []
    merges = []
    replacements = []
    
    num_changed = 0
    
    previous_tokens = sentence_original
    # attack sequences are sorted by divergence and syn-prob -> replace one by one except forbidden
    sentence_tokens_modified = {i:0 for i in range(len(sentence_original))}
    sentence_tokens_modified_by_action = {i:None for i in range(len(sentence_original))}
    
    new_sentence = sentence_obj.__copy__()
    attack_successful = False
    sentences_with_changed_entities = []
    
    for attack in attack_sequences:
                
        idx = attack[0] # token index in the original sentence
        attack_type = attack[1] # type

        if sentence_tokens_modified[idx] == 1:
            continue
        elif sentence_tokens_modified_by_action[max(idx-1,0)] == "insert":
            continue 
        elif attack_type == "insert" and sentence_tokens_modified[idx-1] == 1:
            continue 
        
        original_token = attack[2] # original token
        synonym = attack[4] # synonym
        semantic_sim = attack[5] # semantic similarity
        synonym_prob = attack[6] # probability of the synonym
        divergence = attack[7] # divergence measure
        
        synonym_tokens = bert_tokenizer.tokenize(synonym)
               
        sentence_copy, insertions, merges, replacements, sentence_tokens_modified, sentence_tokens_modified_by_action, shift_idx = __apply_perturb__(attack_type=attack_type, sentence_copy=sentence_copy, synonym_tokens=synonym_tokens,
                                                                                                                                            idx=idx, insertions=insertions, merges=merges, 
                                                                                                                                            replacements=replacements, sentence_tokens_modified=sentence_tokens_modified,
                                                                                                                                            sentence_tokens_modified_by_action=sentence_tokens_modified_by_action)
        new_batch, new_sentence, _, _ = __compute_new_representations__(sentence_obj=new_sentence, previous_tokens=previous_tokens, new_tokens=sentence_copy, pivot_index=shift_idx,attack_type=attack_type, doc_chunking=doc_chunking, chunk=chunk, i=num_changed)
        
        num_changed += 1
        num_queries += 1
        if victim_type == "extraction":
            new_prediction = slot_filling_module.get_start_end_positions(new_batch) 
            
            divergence = __compute_divergence__(original_prediction, new_prediction, victim_type=victim_type)
            
            new_prediction_split = split_prediction_into_sentences(new_prediction, doc_chunking)
            sentences_with_changed_entities, num_entites_changed = compute_entity_changed(original_prediction_split, new_prediction_split, doc_chunking=doc_chunking, 
                                                                                            sentence_object=new_sentence, slot_indices=slot_filling_module.slot_indices, 
                                                                                            original_tokens=sentence_original)
            attack_successful = True if len(sentences_with_changed_entities) > 0 else False

            temp = {
                "sentence_index": sentence_obj.get_index(),
                "token_index": idx, 
                "attack_type": attack_type,
                "original_tokens": original_token,
                "synonym_tokens": synonym,
                "synonym_prob": float(synonym_prob),
                "semantic_sim": float(semantic_sim),
                "divergence": float(divergence),
                "num_changed": num_changed,
                "num_queries": num_queries, 
                "changed_sentences": sentences_with_changed_entities,
                "num_entities_changed": num_entites_changed,
                "comp_mode": None,
                "attack_successful": attack_successful
            }
            attack_logs.append(temp)

        else:
            # batches_comp_scores, batches_comp_scores_no_sigmoid
            comp_scores, comp_scores_no_sigmoid = slot_filling_module.compute_entity_comp_for_attacking(document, doc_chunking, new_batch, sigmoid=False)
            
            new_prediction = comp_scores
            new_prediction_split = comp_scores_no_sigmoid

            attack_successful, num_changes_in_pred, divergence = compute_compatibility_changed(new_pred=new_prediction, original_pred=original_prediction, 
                                                                                                    new_prediction_split=new_prediction_split, original_prediction_split=original_prediction_split, 
                                                                                                    mode=comp_mode, victim_type=victim_type)

            
            temp = {
                "sentence_index": sentence_obj.get_index(), 
                "token_index": idx,
                "attack_type": attack_type,
                "original_tokens": original_token, 
                "synonym_tokens": synonym, 
                "synonyms_prob": float(synonym_prob),
                "semantic_sim": float(semantic_sim),
                "divergence": divergence,
                "num_changes_in_pred": num_changes_in_pred,
                "num_changed": num_changed,
                "num_queries": num_queries,
                "changed_sentences": None,
                "num_entities_changed": None,     
                "comp_mode": comp_mode,      
                "attack_successful": attack_successful, 
            }
            attack_logs.append(temp)

        if attack_successful:
                break
        
        previous_tokens = sentence_copy.copy()

    # iterated over all attack sequences without success
    if not attack_successful:
        new_sentence.set_tokens(sentence_original)
        num_changed = 0
        # append one line to show attack for sentence was not successfull
        temp = {
            "sentence_index": sentence_obj.get_index(),
            "token_index": None, 
            "attack_type": None,
            "original_token": None,
            "synonym": None,
            "synonym_prob": None,
            "semantic_sim": None,
            "divergence": None,
            "num_changed": num_changed,
            "num_queries": num_queries, 
            "attack_successful": False
        }
        attack_logs.append(temp)
        
    return new_sentence, attack_logs, sentences_with_changed_entities

def __apply_perturb__(attack_type, sentence_copy, synonym_tokens, idx, insertions, merges, replacements, sentence_tokens_modified, sentence_tokens_modified_by_action):
    shift_idx = __shift_idx_depending_on_prev_perturb__(idx, insertions, merges)
    # replace the token if replace action selected
    if attack_type == 'replace':
        sentence_copy = sentence_copy[:shift_idx] + synonym_tokens + sentence_copy[shift_idx+1:]            

        # if shift_idx > 0: forbid_merges.add(shift_idx-1) # forbid merging the previous with the newly replaced one
        for i in range(len(synonym_tokens)):
            replacements.append(shift_idx + i)
    
                
    # insert the token if insert action selected
    elif attack_type == 'insert':
        sentence_copy = sentence_copy[:shift_idx] + synonym_tokens + sentence_copy[shift_idx:]

        for i in range(len(synonym_tokens)):
            insertions.append(idx + i)
        
    # merge the token if merge action selected
    elif attack_type == 'merge':

        sentence_copy = sentence_copy[:shift_idx] + synonym_tokens + sentence_copy[shift_idx+2:]
        sentence_tokens_modified[idx+1] = 1 # need to make sure that merged token also modified
        for i in range(len(synonym_tokens)):   
            merges.append(idx + i)

    sentence_tokens_modified[idx] = 1
    sentence_tokens_modified_by_action[idx] = attack_type

    return sentence_copy, insertions, merges, replacements, sentence_tokens_modified, sentence_tokens_modified_by_action, shift_idx

def __shift_idx_depending_on_prev_perturb__(idx, insertions, merges):
    # shift the attack index by insertions history
    shift_idx = idx
    for prev_insert_idx in insertions:
        if idx >= prev_insert_idx:
            shift_idx +=1
    for prev_merge_idx in merges:
        if idx >= prev_merge_idx + 1:
            shift_idx -= 1
    return shift_idx 
                    
def __return_masked_lm_inputs__(sentence_copy, original_sentence, attack_loc):

    mask_inputs, pivot_indices, attack_types, mask_tokens = [], [], [], []
    
    # compute pos-tags for the tokens
    token_tags = nltk.pos_tag(sentence_copy, tagset='universal')
    
    # compute the indices that can be changed
    if attack_loc == 'pos_tag_filter':
        replace_indices, insert_indices, merge_indices = index_from_pos_tag(token_tags)
    #elif attack_loc == 'salience_score':
    #    replace_indices, insert_indices, merge_indices = index_from_salience_score(token_tags, sentence, orig_pred, model)
    #elif attack_loc == 'brute_force':
    #    replace_indices = range(len(token_tags))
    #    insert_indices = range(1, len(token_tags))
    #    merge_indices = find_merge_index(token_tags)

    for replace_idx in replace_indices:
        
        if punct_re.search(sentence_copy[replace_idx]) is not None and \
            words_re.search(sentence_copy[replace_idx]) is None:
            continue
        if sentence_copy[replace_idx].lower() in stop_words_set:
            continue
        # if threshold dict has bool filter_adj, else False -> do not sub adv and adj
        if thres.get('filter_adj', False) and token_tags[replace_idx][1] in ['ADJ', 'ADV']:
            continue
        # keep similarity by not replacing no,not, ...
        if thres.get('keep_sim', False):
            if sentence_copy[replace_idx].lower() in binary_words:
                continue
        
        mask_input = sentence_copy.copy()
        mask_input[replace_idx] = '<mask>'
        mask_inputs.append(" ".join(mask_input).replace(" ##", ""))
        
        orig_token = original_sentence[replace_idx]
        
        mask_tokens.append(orig_token)
        attack_types.append('replace')
        pivot_indices.append(replace_idx)

    # check insertion choices
    for insert_idx in insert_indices:
        if thres.get('keep_sim', False):
            if sentence_copy[insert_idx-1].lower() in binary_words:
                continue
        mask_input = sentence_copy.copy()
        mask_input.insert(insert_idx, '<mask>')
        mask_inputs.append(" ".join(mask_input).replace(" ##", ""))
        mask_tokens.append("")
        attack_types.append('insert')
        pivot_indices.append(insert_idx)
        


    # check merge choices
    for merge_idx in merge_indices:
        if thres.get('keep_sim', False):
            if (sentence_copy[merge_idx].lower() in binary_words or \
                sentence_copy[merge_idx+1].lower() in binary_words):
                continue
            
        mask_input = sentence_copy.copy()
        mask_input[merge_idx] = '<mask>'
        del mask_input[merge_idx+1]
        mask_inputs.append(" ".join(mask_input).replace(" ##", ""))
        orig_token = " ".join([sentence_copy[merge_idx], sentence_copy[merge_idx+1]]).replace(" ##", "")
        mask_tokens.append(orig_token)
        attack_types.append('merge')
        pivot_indices.append(merge_idx)

    return mask_inputs, mask_tokens, attack_types, pivot_indices

def __filter_synonyms_by_thres__(synonyms, sentence_copy, pivot_indices, attack_types):
    len_text = len(sentence_copy) #TODO test with RoBerta Tokenizer here 
    # filter the candidate by syn_probs and synonyms and then query the target models
    synonyms_, syn_probs_, pivot_indices_, attack_types_, new_tokens_roberta_, new_tokens_bert_, orig_tokens_ = [], [], [], [], [], [], []
    for i in range(len(synonyms)):

        attack_type = attack_types[i]
        idx = pivot_indices[i]

        if attack_type == 'replace':
            # iterate over candidates (topk = 5 by default)
            for j in range(len(synonyms[i])):
                if synonyms[i][j]["score"] > thres['replace_prob']:

                    synonym = synonyms[i][j]["token_str"].strip()
                    orig_token = sentence_copy[pivot_indices[i]]
                    if synonym.lower() == orig_token.lower():
                        continue
                    synonyms_.append(synonym)
                    orig_tokens_.append(orig_token)
                    syn_probs_.append(synonyms[i][j]["score"])
                    pivot_indices_.append(idx)
                    attack_types_.append('replace')
                    new_tokens_roberta_.append(sentence_copy[:idx] + [synonym] + sentence_copy[min(idx + 1, len_text):])
                    sentence_string = " ".join((sentence_copy[:idx] + [synonym] + sentence_copy[min(idx + 1, len_text):])).replace(" ##", "")
                    new_tokens_bert_.append(bert_tokenizer.tokenize(sentence_string))

        if attack_type == 'insert':
            for j in range(len(synonyms[i])):
                if synonyms[i][j]["score"] > thres['insert_prob']:
                    synonym = synonyms[i][j]["token_str"].strip()
                    # don't insert punctuation
                    if punct_re.search(synonym) is not None and words_re.search(synonym) is None:
                        continue
                    synonyms_.append(synonym)
                    orig_tokens_.append(sentence_copy[pivot_indices[i]-1])
                    syn_probs_.append(synonyms[i][j]["score"])
                    pivot_indices_.append(idx)
                    attack_types_.append('insert')
                    new_tokens_roberta_.append(sentence_copy[:idx] + [synonym] + sentence_copy[min(idx, len_text):])
                    sentence_string = " ".join(sentence_copy[:idx] + [synonym] + sentence_copy[min(idx, len_text):]).replace(" ##", "")
                    new_tokens_bert_.append(bert_tokenizer.tokenize(sentence_string))

        if attack_type == 'merge':
            for j in range(len(synonyms[i])):
                if synonyms[i][j]["score"] > thres['merge_prob']:
                    synonym = synonyms[i][j]["token_str"].strip()
                    synonyms_.append(synonym)
                    orig_tokens_.append(" ".join(sentence_copy[pivot_indices[i]:pivot_indices[i]+2]))
                    syn_probs_.append(synonyms[i][j]["score"])
                    pivot_indices_.append(idx)
                    attack_types_.append('merge')
                    new_tokens_roberta_.append(sentence_copy[:idx] + [synonym] + sentence_copy[min(idx + 2, len_text):])
                    sentence_string = " ".join(sentence_copy[:idx] + [synonym] + sentence_copy[min(idx + 2, len_text):]).replace(" ##", "")
                    new_tokens_bert_.append(bert_tokenizer.tokenize(sentence_string))

    syn_probs = np.array(syn_probs_)

    return synonyms_, pivot_indices_, attack_types_, new_tokens_roberta_, new_tokens_bert_, orig_tokens_, syn_probs_
    
def __compute_new_representations__(sentence_obj, previous_tokens, new_tokens, pivot_index, attack_type, doc_chunking, chunk, i):
        
        # Recompute objects with new tokens, entities etc.
        new_sentence = sentence_obj.__deepcopy__(memo={}) 
        new_sentence.set_tokens(new_tokens)

        update_entities_CLARE(new_sentence, original_tokens=previous_tokens, new_sentence_tokens=new_tokens, idx=pivot_index, attack_type=attack_type)

        old_doc_chunking = doc_chunking.__deepcopy__(memo={})
        old_chunk = chunk.__deepcopy__(memo={})

        old_sentence_offsets = doc_chunking.get_sentence_offsets(flattened=False)
        chunk.set_sentence_by_index(new_sentence.get_index(), new_sentence)
        chunk.update_sentence_offsets() 
        
        chunk_index = chunk.get_chunk_index()

        doc_chunking.set_chunk_by_index(chunk_index=chunk_index, new_chunk=chunk, sentence_index=new_sentence.get_index())
        doc_chunking.update_sentence_offsets()
        # doc_chunking.set_from_chunks(chunk, sentence_obj.get_index()) 
        new_sentence_offsets = doc_chunking.get_sentence_offsets(flattened=False)
        
        # compute batch from doc_chunking again
        new_batch = create_batches_from_chunking([doc_chunking], document_encoder, slot_filling_module.slot_indices)
        return new_batch, new_sentence, old_chunk, old_doc_chunking

def __reset_to_original__(doc_chunking_original, chunk_original, list_of_original_chunks, list_of_original_sentences):
    # print("in reset original")
    c_index = chunk_original.get_chunk_index()
    doc_chunking = doc_chunking_original
    doc_chunking.set_all_chunks(list_of_original_chunks)
    doc_chunking.update_sentence_offsets()

    for i, c in enumerate(doc_chunking.get_chunks()):
        c.set_sentences(list_of_original_sentences[i])
    chunk = doc_chunking.get_chunks()[c_index]
    return doc_chunking, chunk

def __compute_divergence_for_all_sequences__(document, doc_chunking, chunk, sentence_obj, sentence_original, original_prediction, original_prediction_split, semantic_sims, new_tokens_bert, pivot_indices, attack_types, orig_tokens, synonyms, syn_probs):
    
    synonyms_, syn_probs_, pivot_indices_, attack_types_, new_tokens_bert_, semantic_sims_, orig_tokens_, divergences_ = [], [], [], [], [], [], [], []
    doc_chunking_original = doc_chunking.__deepcopy__(memo={})
    chunk_original = chunk.__deepcopy__(memo={})
    list_of_original_chunks = [c.__deepcopy__(memo={}) for c in doc_chunking.get_chunks()]
    list_of_original_sentences = [] 
    for c in doc_chunking.get_chunks():
        list_of_original_sentences.append(c.get_sentences())
    
    # iterate over changes and test iteratively -> get kl divergence for each single change
    for i in range(len(semantic_sims)):
        
        # skip if semantic similarity is too low
        if semantic_sims[i] < thres[f"{attack_types[i]}_sim"]:
            continue
        
            
        # get the original token (TODO: check if matches)
        orig_token = orig_tokens[i]
        
        if attack_types[i] == "replace":
            should_be = sentence_obj.get_tokens()[pivot_indices[i]]
        elif attack_types[i] == "merge":
            should_be = " ".join(sentence_obj.get_tokens()[pivot_indices[i]:pivot_indices[i]+2])
        elif attack_types[i] == "insert":
            should_be = sentence_obj.get_tokens()[pivot_indices[i]-1]
        
        try:
            assert should_be == orig_token, "orig_token is not what it should be"
        
        except AssertionError:
            print(f"{orig_token} != {should_be}")
        new_batch, _, old_chunk, old_doc_chunking = __compute_new_representations__(sentence_obj, previous_tokens=sentence_original, new_tokens=new_tokens_bert[i], 
                                                                                    pivot_index=pivot_indices[i], attack_type=attack_types[i], doc_chunking=doc_chunking, chunk=chunk, i=i)

        if victim_type == "extraction":
            new_prediction = slot_filling_module.get_start_end_positions(new_batch) 
            


        else:
            _, comp_scores_no_sigmoid = slot_filling_module.compute_entity_comp_for_attacking(document, doc_chunking, new_batch, sigmoid=False)
            new_prediction = comp_scores_no_sigmoid

        divergence = __compute_divergence__(original_prediction=original_prediction, new_prediction=new_prediction, victim_type=victim_type)

        doc_chunking, chunk = __reset_to_original__(doc_chunking_original=doc_chunking_original, chunk_original=chunk_original, 
                                                        list_of_original_chunks=list_of_original_chunks, 
                                                        list_of_original_sentences=list_of_original_sentences)

        if abs(divergence) < thres[f"divergence_thres_{victim_type}"]:
            continue

        synonyms_.append(synonyms[i])
        syn_probs_.append(syn_probs[i])
        pivot_indices_.append(pivot_indices[i])
        attack_types_.append(attack_types[i])
        new_tokens_bert_.append(new_tokens_bert[i])
        semantic_sims_.append(semantic_sims[i])
        orig_tokens_.append(orig_token)
        divergences_.append(divergence)

    return synonyms_, syn_probs_, pivot_indices_, attack_types_, new_tokens_bert_, semantic_sims_, orig_tokens_, divergences_

def __sort_by_divergence__(collections):

    # for each choice, find the best attack choices
    attack_sequences = []
    # sorts by divergences and if identical by syn_prob 
    collections.sort(key=lambda x:(x[-1],x[-2]), reverse=True)
    highest_divergence = collections[0][-1] 
    best_sequence = collections[0]

    for sequence in collections:
        # if new choice appear
        if best_sequence[:2] != sequence[:2]:
            attack_sequences.append(best_sequence)
            best_sequence = sequence
            highest_divergence = sequence[-1]
            continue
        if sequence[-1] > highest_divergence: # TODO: eval if < or > 
            highest_divergence = sequence[-1]
            best_sequence = sequence

    attack_sequences.append(best_sequence)
    attack_sequences.sort(key=lambda x : (x[-1], x[-2]), reverse=True)
    
    return attack_sequences

def get_attack_sequences(sentence, sentence_obj, chunk, doc_chunking, document, original_prediction, original_prediction_split, attack_loc="pos_tag_filter"):
    sentence_copy = sentence.copy()
    sentence_original = deepcopy(sentence)
    
    mask_inputs, mask_tokens, attack_types, pivot_indices = __return_masked_lm_inputs__(sentence_copy=sentence_copy, original_sentence=sentence_original, attack_loc=attack_loc)

    if len(mask_inputs) == 0:
        return [], 0
    
    synonyms = roberta_mlm(mask_inputs, mask_tokens, top_k=1)
    if len(synonyms) == 0:
        return [], 0
    
    synonyms, pivot_indices, attack_types, new_tokens_roberta, new_tokens_bert, orig_tokens, syn_probs = __filter_synonyms_by_thres__(synonyms, sentence_copy, pivot_indices, attack_types)

    semantic_sims = similarity_calculation(indices=pivot_indices, orig_texts=[sentence_copy] * len(new_tokens_bert), new_texts=new_tokens_bert, sim_predictor=use_sim, thres=thres, attack_types=attack_types)

        
    synonyms, syn_probs, pivot_indices, attack_types, new_tokens_bert, semantic_sims, orig_tokens, divergences = __compute_divergence_for_all_sequences__(document=document, doc_chunking=doc_chunking, chunk=chunk, sentence_obj=sentence_obj, 
                                                                                                                        semantic_sims=semantic_sims, new_tokens_bert=new_tokens_bert, 
                                                                                                                        pivot_indices=pivot_indices, attack_types=attack_types, 
                                                                                                                        orig_tokens=orig_tokens, synonyms=synonyms, syn_probs=syn_probs,
                                                                                                                        sentence_original=sentence_original, original_prediction=original_prediction, 
                                                                                                                        original_prediction_split=original_prediction_split)

    if len(new_tokens_bert) == 0:
        return [], 0
    
    # zip all lists together -> each elem is one attack
    collections = [list(c) for c in list(zip(pivot_indices, attack_types, orig_tokens, new_tokens_bert, synonyms, semantic_sims, syn_probs, divergences))]
    num_query = len(new_tokens_bert)
    attack_sequences = __sort_by_divergence__(collections)    
    
    return attack_sequences, num_query

def index_from_pos_tag(token_tags):
    replace_loc, insert_loc, merge_loc = [], [], []
    for idx in range(len(token_tags)):
        if token_tags[idx][1] in pos_tag_filter['replace']:
            replace_loc.append(idx)
        if idx > 0 and "%s/%s" % (token_tags[idx-1][1], token_tags[idx][1]) \
            in pos_tag_filter['insert']:
            insert_loc.append(idx)
        if idx < len(token_tags) - 1 and \
            "%s/%s" % (token_tags[idx][1], token_tags[idx+1][1]) in pos_tag_filter['merge']:
            merge_loc.append(idx)
    return replace_loc, insert_loc, merge_loc
               


def run_attack(slot_filling_module, comp_mode, model="CLARE", victim_type="comp", start=0, end=len(slot_filling_module.documents_train)-1, data_mode="train"):
    output = []
    attacked_doc_chunkings = []
    attacked_documents = []
    attacked_batches = []
    attacked_sentences = []
    
    #import random
    #zipped_list = list(zip(slot_filling_module.document_chunkings_train, slot_filling_module.documents_train, slot_filling_module.batches_train))
    #random.shuffle(zipped_list)
    #slot_filling_module.document_chunkings_train, slot_filling_module.documents_train, slot_filling_module.batches_train = zip(*zipped_list)
    
    
    documents, document_chunkings, batches = get_data_depending_on_split(slot_filling_module,data_mode=data_mode)
    
    assert start >= 0
    assert end <= len(documents)
    print(f"working doc {start}-{end} from {data_mode} set with {len(documents)} documents in total // (disease {disease_str})")
    
    for dc_index in tqdm(range(start, end+1)):
        
        # document
        document = documents[dc_index]
        document_modified = document.__deepcopy__(memo={})
        
        abstract_id = document._abstract.abstract_id

        print(f"Doc chunking: {dc_index} and abstract id {abstract_id}")
        
        batch = batches[dc_index]

        doc_chunking = document_chunkings[dc_index]

        list_of_original_sentences = [] 
        for c in doc_chunking.get_chunks():
            list_of_original_sentences.append([(s.__deepcopy__(memo={})) for s in c.get_sentences()])
    
        doc_chunking_original = doc_chunking.__deepcopy__(memo={})
        doc_chunking_modified = doc_chunking.__deepcopy__(memo={})

        output_doc_chunking = []
        attacked_sentences = [] 

        if victim_type == "extraction":
            original_prediction = slot_filling_module.get_start_end_positions([batch]) # shape(#batches, 2, 512)
            original_prediction_split = split_prediction_into_sentences(original_prediction, doc_chunking)
        else:
            # batches_comp_scores, batches_comp_scores_no_sigmoid
            comp_scores, comp_scores_no_sigmoid = slot_filling_module.compute_entity_comp_for_attacking(document, doc_chunking, [batch], sigmoid=False)
            
            original_prediction = comp_scores
            original_prediction_split = comp_scores_no_sigmoid
        
        # iterate over chunks
        for c_index in range(len(doc_chunking_original.get_chunks())):
            print(f"Chunk: {c_index}")
            chunk = doc_chunking.get_chunks()[c_index]
            chunk_modified = chunk.__deepcopy__(memo={})
            
            # iterate over sentences in chunk
            for s_index in chunk.get_sentence_indices():
                
                if s_index in attacked_sentences:
                    continue

                sentence = chunk.get_sentence_by_index(s_index)
                sentence_original = sentence.__deepcopy__(memo={})
                original_entities = [e.__deepcopy__(memo={}) for e in sentence_original.get_entities()]

                old_tokens = sentence.get_tokens()

                ### CLARE specific 
                new_sentence, attack_logs, changed_sentences = apply_clare(document=document, doc_chunking=doc_chunking, batch=batch, chunk=chunk, 
                                                                    sentence_obj=sentence, victim_type=victim_type, 
                                                                    original_prediction=original_prediction, original_prediction_split=original_prediction_split)

            
                if victim_type == "extraction":
                    print_attack_logs(attack_logs, s_index, new_sentence=new_sentence.get_tokens(), old_sentence=old_tokens, changed_sentences=changed_sentences)
                elif victim_type == "comp":
                    print_attack_logs_comp(attack_logs, s_index, new_sentence=new_sentence.get_tokens(), old_sentence=old_tokens)
                
                else:
                    print(f"{victim_type.upper()} NOT A VALID VICTIM")
                    exit() 
            
                # if the attack was not successful -> reset entities to original
                if not attack_logs[-1]["attack_successful"]:
                    new_sentence.set_entities(original_entities)
                    assert new_sentence.get_entities()[0] == original_entities[0]

                # reset doc_chunking
                doc_chunking = doc_chunking_original
                #original_chunks = [Chunk(sentences=values, chunk_index=ind) for ind, values in enumerate(list_of_original_sentences)]
                #doc_chunking.set_all_chunks(original_chunks)
                sentence_original.set_entities(original_entities)
                doc_chunking.get_chunks()[c_index].set_sentence_by_index(sentence_index=s_index, new_sentence=sentence_original)
                assert doc_chunking == doc_chunking_original, "Doc chunking not same as original doc chunking"
                assert doc_chunking.get_chunks()[c_index].get_sentence_by_index(s_index) == sentence_original, "Sentence in doc chunking not original_sentence"
                assert doc_chunking.get_chunks()[c_index].get_sentence_by_index(s_index).get_entities() == original_entities, "Entities in doc_chunk not original entities"
                
                # reset chunk
                chunk = doc_chunking.get_chunks()[c_index]
                chunk.update_sentence_offsets()
                assert chunk.get_sentence_by_index(s_index) == sentence_original, "Sentence in chunk not original sentence"
                assert chunk.get_sentence_by_index(s_index).get_entities() == original_entities, "Entities in chunk not original entities"
                
                # store the modified sentence in the modified_chunk
                chunk_modified.set_sentence_by_index(sentence_index=s_index, new_sentence=new_sentence)
                attack_dict = {s_index:attack_logs}
                output_doc_chunking.append(attack_dict)
                attacked_sentences.append(s_index)
            
            # done with one chunk from doc_chunking -> set the modified chunk
            # doc_chunking_modified.set_from_chunks(chunk_modified, 0)
            doc_chunking_modified.set_chunk_by_index(chunk_index=chunk_modified.get_chunk_index(), new_chunk=chunk_modified, sentence_index=new_sentence.get_index())
            doc_chunking_modified.update_sentence_offsets() 
            doc_chunking = doc_chunking_original
        
        # compute batch from doc_chunking again
        attacked_batch = create_batches_from_chunking([doc_chunking_modified], document_encoder, slot_filling_module.slot_indices)
        attacked_doc_chunkings.append(doc_chunking_modified)
        attacked_batches.append(attacked_batch)
        document_modified.set_sentences(doc_chunking_modified.get_sentences())
        attacked_documents.append(document_modified)

        output.append({abstract_id:output_doc_chunking})


    print("Saving modified data")
    save_attack_logs_json(attack_logs=output, disease_str=disease_str, model=model, victim_type=victim_type, 
                                data_mode=data_mode, start=start, end=end, comp_mode=comp_mode)
    save_modified_data(attacked_doc_chunkings=attacked_doc_chunkings, attacked_batches=attacked_batches, attacked_documents=attacked_documents, 
                        model=model, disease_str=disease_str, victim_type=victim_type, comp_mode=comp_mode, data_mode=data_mode, 
                        start=start, end=end)

if __name__ == "__main__":
    from tqdm import tqdm
    model = "CLARE"

    # if nothing passed: set to 0 and last document
    if not start:
        start = 0
    if not end and data_mode == "train":
        end = len(slot_filling_module.documents_train)-1
    elif not end and data_mode == "test":
        end = len(slot_filling_module.documents_test)-1
    
    run_attack(slot_filling_module, model="CLARE", victim_type=victim_type, start=start, end=end, comp_mode=comp_mode, data_mode=data_mode)
    # run_attack(slot_filling_module, model="CLARE", victim_type=victim_type, start=half_of_abstracts+1, end=len(slot_filling_module.documents_train)-1)

    