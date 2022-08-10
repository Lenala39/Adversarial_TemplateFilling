import difflib
from operator import sub 
import os
import spacy_alignments 
import tensorflow as tf
import pickle 
import argparse
import numpy as np
import json

kl = tf.losses.KLDivergence(tf.keras.losses.Reduction.NONE)
dir_path = os.path.dirname(os.path.realpath(__file__)) 

class bcolors:
    SAME = '\033[37m' #WHITE
    REMOVE = '\033[91m' #RED
    BLUE = '\033[34m' #BLUE
    INSERT = '\033[32m' #GREEN
    RESET = '\033[0m' #RESET COLOR

def pretty_print_difference(original, modified):

    diff = difflib.ndiff(original, modified)
    output_string = ""
    for elem in diff:
        if "  " in elem:
            # both sequences
            output_string += f"{bcolors.SAME}{elem[2:]} {bcolors.RESET}"
        elif "+ " in elem:
            # only in modified -> inserted
            output_string += f"{bcolors.INSERT}{elem[2:]} {bcolors.RESET}"
        elif "- " in elem:
            # only in original -> deleted
            output_string += f"{bcolors.REMOVE}{elem[2:]} {bcolors.RESET}"
        elif "? " in elem:
            # in no sequence??
            pass 

    print(output_string)
            

def split_prediction_into_sentences(prediction, doc_chunking):
    
    start_pred = tf.argmax(prediction[0], axis=-1)[0]
    end_pred = tf.argmax(prediction[1], axis=-1)[0]
    
    start_predictions_split = __split_prediction_into_sentences__(start_pred, doc_chunking)
    end_predictions_split = __split_prediction_into_sentences__(end_pred, doc_chunking)

    return [start_predictions_split, end_predictions_split]


def __split_prediction_into_sentences__(pred, doc_chunking):
    predictions_split = []
    for chunk_index in range(len(doc_chunking.get_chunks())):
        # prediction from chunk (512)
        chunk_pred = pred[chunk_index]
        # chunk object
        chunk = doc_chunking.get_chunks()[chunk_index]
        chunk_prediction_split = []
        sentence_indices = list(chunk.get_sentence_indices())
        sentence_indices.sort()
        for i in range(len(sentence_indices)):
            sentence_index = sentence_indices[i] 
            offset_start = chunk.get_sentence_offset(sentence_index)
            try:
                offset_end = chunk.get_sentence_offset(sentence_indices[i+1])
            except IndexError:
                offset_end = offset_start + chunk.get_sentence_by_index(sentence_index).get_num_tokens() + 1
            
            # last sentence 
            if not offset_end and sentence_index == sentence_indices[-1]:
                offset_end = offset_start + chunk.get_sentence_by_index(sentence_index).get_num_tokens() + 1 # +1 because last sentence has no [SEP at the end]

            temp = chunk_pred[offset_start:offset_end-1] # -1 because next sentence start minus [SEP]
            try:
                assert temp.shape[0] == chunk.get_sentence_by_index(sentence_index).get_num_tokens(), "shape of sentence does not match extracted temp from pred"
            except AssertionError:
                print(f"ERROR in asserting sentence length of sentence {sentence_index}")
                print(f"temp has len {temp.shape[0]}")
                print(f"sentence has len {chunk.get_sentence_by_index(sentence_index).get_num_tokens()}")

            chunk_prediction_split.append(temp)
        predictions_split.append(chunk_prediction_split)
        
    return predictions_split

def __get_chunk_with_sentence(sentence_index, doc_chunking):
     # get the correct chunk that includes the sentence
    for c in doc_chunking.get_chunks():
        if sentence_index in c._sentence_offsets:
            chunk = c
            chunk_index = c.get_chunk_index()
            break 

    #position in list vs sentence index (some sentences are skipped)
    mapped_sentence_index = chunk.get_mapping_sentence_index_to_list_index(sentence_index) 

    if chunk_index > 0:
        for c in doc_chunking.get_chunks():
            if c.get_chunk_index() < chunk_index:
                mapped_sentence_index += c.get_num_sentences()

    return mapped_sentence_index

def __compare_unchanged_sentences(mapped_sentence_index, start_pred_new, start_pred_original, end_pred_new, end_pred_original):
    changed_sentence_indices = []
    for i in range(len(start_pred_new)):
        if i != mapped_sentence_index:    
            try:
                comparison_start = tf.math.equal(start_pred_new[i],start_pred_original[i])
                comparison_end = tf.math.equal(end_pred_new[i],end_pred_original[i])
            except Exception as e:
                print(e)
                continue
            if not tf.math.reduce_all(comparison_start).numpy() and not tf.math.reduce_all(comparison_end):
                true_elems_start = tf.math.count_nonzero(comparison_start).numpy() # counts True -> 
                false_elems_start = comparison_start.shape[0] - true_elems_start
                true_elems_end = tf.math.count_nonzero(comparison_end).numpy() # counts True -> 
                false_elems_end = comparison_end.shape[0] - true_elems_end

                changed_sentence_indices.append((i, (false_elems_start + false_elems_end)))
    
    return changed_sentence_indices


def __compare_changed_sentence_spacy(sentence_object, mapped_sentence_index, start_pred_new, end_pred_new, start_pred_original, end_pred_original, original_tokens):
    new2original, original2new = spacy_alignments.get_alignments(sentence_object.get_tokens(), original_tokens)
    missed_entities = 0
    added_entities = 0
    changed_entities = 0

    if new2original == original2new:
        return missed_entities, added_entities, changed_entities

    for new_i in range(len(start_pred_new)):
        new_token_start = start_pred_new[new_i]
        new_token_end = end_pred_new[new_i]
        try:
            original_i = new2original[new_i]
        except IndexError:
            added_entities += 1
            continue 
        
        # if no adequate mapping for original was found -> check that no entites were added
        if len(original_i) == 0 and new_token_start == 0 and new_token_end == 0:
            continue 
        elif len(original_i) == 0 and (new_token_start != 0 or new_token_end != 0):
            added_entities += 1
            continue 
        elif len(original_i) > 1:
            missed_entities += 1
            continue
        elif len(original_i) == 0:
            added_entities += 1
            continue

        original_token_start = start_pred_original[original_i[0]:original_i[-1]+1]
        original_token_end = end_pred_original[original_i[0]:original_i[-1]+1]

        # all match
        if new_token_start == original_token_start and new_token_end == original_token_end:
            continue 
        
        elif (new_token_start == 0 and original_token_start != 0) or (new_token_end == 0 and original_token_end != 0):
            missed_entities += 1
        
        elif (new_token_start != 0 and original_token_start == 0) or (new_token_end != 0 and original_token_end == 0):
            added_entities += 1

        elif (new_token_start != 0 and original_token_start != 0 and new_token_start != original_token_start) or ((new_token_end != 0 and original_token_end != 0 and new_token_end != original_token_end)):
            changed_entities += 1

    return missed_entities, added_entities, changed_entities



def __map_sentence_indices(changed_sentence_indices, doc_chunking):

    # get the size of all chunks
    chunk_sizes = []
    for c in doc_chunking.get_chunks():
        sentence_indices = list(c._sentence_offsets.keys())
        chunk_sizes.append((sentence_indices, c.get_chunk_index()))

    changed_sentence_indices_mapped = []
    # iterate over sentences that had changes: map back to their chunks
    for idx, _ in changed_sentence_indices:
        # check what chunk a sentence belongs to
        for s_indices, c_index in chunk_sizes:
            if idx+1 in s_indices:
                chunk_index_for_mapping = c_index
                
                if chunk_index_for_mapping > 0:
                    for s_indices, c_index in chunk_sizes:
                        if c_index < chunk_index_for_mapping:
                            idx = idx - len(s_indices) - 1 # substract size of first chunk -> want 0, 1, 2, ... for new chunk (not 14, ...)
                    
                new_idx = doc_chunking.get_chunks()[chunk_index_for_mapping].get_mapping_list_index_to_sentence_index(idx)
                changed_sentence_indices_mapped.append(new_idx)
                break
    return changed_sentence_indices_mapped

def compute_entity_changed(original_prediction, new_prediction, doc_chunking, sentence_object, slot_indices, original_tokens):
    
    changed_sentence_indices = []

    sentence_index = sentence_object.get_index()

    # flatten chunks so that it does not matter for comparison if sentence had to be    
    start_pred_original = [sentence for chunk in original_prediction[0] for sentence in chunk]   
    end_pred_original = [sentence for chunk in original_prediction[1] for sentence in chunk]   
    
    start_pred_new = [sentence for chunk in new_prediction[0] for sentence in chunk]   
    end_pred_new = [sentence for chunk in new_prediction[1] for sentence in chunk]   
    
    mapped_sentence_index = __get_chunk_with_sentence(sentence_index, doc_chunking)
   
    changed_sentence_indices = __compare_unchanged_sentences(mapped_sentence_index, start_pred_new, start_pred_original, 
                                                                end_pred_new, end_pred_original)

    try:
        assert start_pred_new[mapped_sentence_index].shape[0] == sentence_object.get_num_tokens(), "shape of sentence does not match extracted temp from pred"
    except AssertionError:
        print("Problem in compute entity changed: Assertion for shape failed!")
        print(mapped_sentence_index)
    missed_entities, added_entities, changed_entities = __compare_changed_sentence_spacy(sentence_object, mapped_sentence_index, start_pred_new=start_pred_new[mapped_sentence_index], end_pred_new=end_pred_new[mapped_sentence_index], 
                                        start_pred_original=start_pred_original[mapped_sentence_index], end_pred_original=end_pred_original[mapped_sentence_index], 
                                        original_tokens=original_tokens)
    # changed_sentence_indices_temp, added_entities, missed_entities = __compare_changed_sentence(sentence_object, mapped_sentence_index, start_pred_new, end_pred_new, slot_indices)
    all_changed_entities = missed_entities + added_entities + changed_entities
    if all_changed_entities > 0:
        changed_sentence_indices.append((sentence_object.get_index(), all_changed_entities))

    changed_sentence_indices_mapped = __map_sentence_indices(changed_sentence_indices, doc_chunking)

    return changed_sentence_indices_mapped, all_changed_entities

def __compute_compatibility_changed_class_change__(new_match_scores, old_match_scores):
    MATCHES = 0
    DIFFERENT = 1
    element_wise_comparison_result = []
        
    for b in range(len(new_match_scores)):
        
        num_new_none = len([elem for elem in new_match_scores[b] if not elem])
        num_old_none = len([elem for elem in old_match_scores[b] if not elem])
        if not num_new_none == num_old_none:
            for i in range(len(new_match_scores[b])):
                if not new_match_scores[b][i] and old_match_scores[b][i]:
                    element_wise_comparison_result.append(len(old_match_scores[b][i]))
            continue

        max_list_len = max(max([len(elem) for elem in new_match_scores[b] if elem]), max([len(elem) for elem in old_match_scores[b] if elem]))
        
        # remove all None elems if None in both lists
        old_temp = [elem for i, elem in enumerate(old_match_scores[b]) if elem and new_match_scores[b][i]]
        new_temp = [elem for i, elem in enumerate(new_match_scores[b]) if elem and old_match_scores[b][i]]
        old_temp = __fill_comp_predictions__(max_list_len, old_temp)
        new_temp = __fill_comp_predictions__(max_list_len, new_temp)
            
        old_temp_ = []
        new_temp_ = []
        for i in range(len(old_temp)):
            old_temp_.append([1 if elem > 0.5 else 0 for elem in old_temp[i]])
            new_temp_.append([1 if elem > 0.5 else 0 for elem in new_temp[i]])

        old_temp = old_temp_
        new_temp = new_temp_

        for i in range(len(new_temp)):
            
            if (not new_temp[i] and old_temp[i]) or (new_temp[i] and not old_temp[i]):
                element_wise_comparison_result.append(DIFFERENT)
                continue 
            
            comparison = tf.math.equal(new_temp[i],old_temp[i])
            difference = len(comparison) - tf.math.count_nonzero(comparison).numpy()
            element_wise_comparison_result.append(difference)
            

    if sum(element_wise_comparison_result) > 0:
        return True, sum(element_wise_comparison_result) 
    else:
        return False, 0 

def __compute_compatibility_changed_value_change__(new_prediction, original_prediction):

    changed = False 
    num_changes = 0
    for b in range(len(original_prediction)): # iterate over batches (usually only one)
        num_new_none = len([elem for elem in new_prediction[b] if not elem])
        num_old_none = len([elem for elem in original_prediction[b] if not elem])
        if not num_new_none == num_old_none:
            for i in range(len(new_prediction[b])):
                if not new_prediction[b][i] and original_prediction[b][i]:
                    num_changes += len(original_prediction[b][i])
                    changed = True
            continue

        max_list_len = max(max([len(elem) for elem in new_prediction[b] if elem]), max([len(elem) for elem in original_prediction[b] if elem]))
        
        # remove all None elems if None in both lists
        original_temp = [elem for i, elem in enumerate(original_prediction[b]) if elem and new_prediction[b][i]]
        new_temp = [elem for i, elem in enumerate(new_prediction[b]) if elem and original_prediction[b][i]]
        # pad remaining to all same length
        original_temp = __fill_comp_predictions__(max_list_len, original_temp)
        new_temp = __fill_comp_predictions__(max_list_len, new_temp)
        
        # element wise subtraction 
        subtraction_result = np.subtract(original_temp, new_temp)
        # check if more than 0.5 difference
        filtered_subtraction = np.where(abs(subtraction_result) < 0.5, None, subtraction_result)
        
        if np.count_nonzero(filtered_subtraction) > 0:
            changed = True
            num_changes += np.count_nonzero(filtered_subtraction)
            
    return changed, num_changes

def compute_compatibility_changed(new_pred, original_pred, new_prediction_split, original_prediction_split, mode, victim_type):
    divergence = None 
    num_changes = None 
    
    if mode == "divergence":
        attack_successful, divergence = __compute_compatibility_changed_divergence__(original_prediction=original_prediction_split, 
                                                                                        new_prediction=new_prediction_split, victim_type=victim_type)
        divergence = float(divergence)
    
    elif mode == "class_change":
        attack_successful, num_changes = __compute_compatibility_changed_class_change__(new_match_scores=new_pred, old_match_scores=original_pred)
        divergence = float(__compute_divergence__(original_prediction=original_prediction_split, new_prediction=new_prediction_split, victim_type=victim_type))
        num_changes = int(num_changes)

    elif mode == "value_change":
        attack_successful, num_changes = __compute_compatibility_changed_value_change__(new_prediction=new_pred, original_prediction=original_pred)
        divergence = float(__compute_divergence__(original_prediction=original_prediction_split, new_prediction=new_prediction_split, victim_type=victim_type))
        num_changes = int(num_changes)

    return attack_successful, num_changes, divergence

def __compute_compatibility_changed_divergence__(original_prediction, new_prediction, victim_type):
    divergence = __compute_divergence__(original_prediction, new_prediction, victim_type=victim_type)
    if divergence > 0.2:
        attack_successful = True
    else: 
        attack_successful = False
    
    return attack_successful, divergence


def __compute_divergence__(original_prediction, new_prediction, victim_type):
    # calculate KL divergence as a scoring method
    # check if attack was successful -> use numpy() not tensor as input!
    if victim_type == "extraction":
        new_prediction_start = tf.reshape(new_prediction[0][0], [new_prediction[0].shape[1]*new_prediction[0].shape[2],new_prediction[0].shape[-1]])
        original_prediction_start = tf.reshape(original_prediction[0][0], [original_prediction[0].shape[1]*original_prediction[0].shape[2],new_prediction[0].shape[-1]])
        pad = tf.zeros([new_prediction_start.shape[0]-original_prediction_start.shape[0], new_prediction[0].shape[-1]])
        original_prediction_start = tf.concat([original_prediction_start, pad], axis=0)

        new_prediction_end = tf.reshape(new_prediction[1][0], [new_prediction[1].shape[1]*new_prediction[1].shape[2],new_prediction[0].shape[-1]])
        original_prediction_end = tf.reshape(original_prediction[1][0], [original_prediction[1].shape[1]*original_prediction[1].shape[2],new_prediction[0].shape[-1]])
        pad = tf.zeros([new_prediction_end.shape[0]-original_prediction_end.shape[0], new_prediction[0].shape[-1]])
        original_prediction_end = tf.concat([original_prediction_end, pad], axis=0)

    
        divergence_start = kl(original_prediction_start.numpy(), new_prediction_start.numpy()).numpy()
        divergence_end = kl(original_prediction_end.numpy(), new_prediction_end.numpy()).numpy()
        divergence = (divergence_end.mean() + divergence_start.mean()) / 2.0

    else:
        original_prediction_filled = []
        new_prediction_filled = []
        
        for b in range(len(original_prediction)):
            
            max_list_len = max(max([len(elem) for elem in new_prediction[b] if elem]), max([len(elem) for elem in original_prediction[b] if elem]))
            # remove all None elems if None in both lists
            original_temp = [elem for i, elem in enumerate(original_prediction[b]) if elem and new_prediction[b][i]]
            new_temp = [elem for i, elem in enumerate(new_prediction[b]) if elem and original_prediction[b][i]]

            original_temp = __fill_comp_predictions__(max_list_len, original_temp)
            new_temp = __fill_comp_predictions__(max_list_len, new_temp)
            
            original_temp = tf.nn.softmax(original_temp)
            new_temp = tf.nn.softmax(new_temp)

            original_prediction_filled.append(original_temp)
            new_prediction_filled.append(new_temp)
    
        divergence = kl(original_prediction_filled, new_prediction_filled).numpy().mean()

    return divergence 

def get_data_depending_on_split(slot_filling_module, data_mode="train"):
    if data_mode == "train":
        documents = slot_filling_module.documents_train
        document_chunkings = slot_filling_module.document_chunkings_train
        batches = slot_filling_module.batches_train
    elif data_mode == "test":
        documents = slot_filling_module.documents_test
        document_chunkings = slot_filling_module.document_chunkings_test
        batches = slot_filling_module.batches_test
    
    return documents, document_chunkings, batches


def parse_cli_arguments(for_slot_filling=False):
    parser = argparse.ArgumentParser(description='Processing the input config for the computation.')
    parser.add_argument("-d", "--disease", required=True, help="Which dataset should be attacked: dm2 or gl?", choices=["dm2", "gl"], metavar="DISEASE")
    parser.add_argument("-v", "--victim", required=True, help="Which task should be attacked: extraction or comp?", choices=["extraction", "comp"], metavar="VICTIM")
    parser.add_argument("-s", "--start", required=False, help="start index")
    parser.add_argument("-e", "--end", required=False, help="end index")
    parser.add_argument("-c", "--comp_mode", required=False, help="mode of change for comp task", choices=["divergence", "class_change", "value_change"])
    parser.add_argument("-t", "--train_or_test", required=True, help="Should train or test be attacked", choices=["train", "test"])
    if for_slot_filling:
        parser.add_argument("-m", "--model", required=True, help="Which attack model should be used", choices=["CLARE", "Araujo"])
    # parser.add_argument("-v", "--verbose", required=False, help="should sentences be printed after attack", action="store_false")
    args = vars(parser.parse_args())

    if args["victim"] == "comp" and not args["comp_mode"]:
        print("need to provide a mode for how the change will be computed between the compatibilities.")
        print("options are: divergence, class_change and value_change")
        exit()
    
    start = None 
    end = None 
    # cast to int but check if None 
    if args["start"]:
        start = int(args["start"])
    if args["end"]:
        end = int(args["end"])
    
    if for_slot_filling:
        return args["disease"], args["victim"], start, end, args["comp_mode"], args["train_or_test"], args["model"]
    else:
        return args["disease"], args["victim"], start, end, args["comp_mode"], args["train_or_test"]


def __get_file_descr_with_params__(model, disease_str, victim_type, comp_mode, data_mode, start, end):
     # start and end pos into one string
    if not start and not end:
        postfix = "all"
    else:
        postfix = f"{start}to{end}"

    # comp mode
    if victim_type.lower() == "comp" and not comp_mode:
        comp_mode = "XXXCompMode_"
    elif victim_type.lower() == "extraction":
        comp_mode = ""
    else:
        comp_mode = comp_mode + "_"

    return f"{model}_{disease_str}_{victim_type}_{data_mode}_{comp_mode}{postfix}"

def save_attack_logs_json(attack_logs,disease_str, model, victim_type, data_mode, start=None, end=None, comp_mode=None):
    filename_identifier = __get_file_descr_with_params__(model=model, disease_str=disease_str, victim_type=victim_type, start=start, end=end, comp_mode=comp_mode, data_mode=data_mode)
    
    filename = os.path.join(dir_path, "attack_logs/" f"{filename_identifier}_results.json")
    json_string = json.dumps(attack_logs)
    
    print(f"Saving attack logs to {filename}")
    
    with open(filename, "w") as json_out:
        json_out.write(json_string)

def save_modified_data(attacked_doc_chunkings, attacked_batches, attacked_documents, disease_str, model, victim_type, data_mode, further_descr=None, start=None, end=None, comp_mode=None, subfolder="Attacked"):
    """
    Saves the attacked doc_chunkings and batches as pickle
    
    """
    filename_identifier = __get_file_descr_with_params__(model=model, disease_str=disease_str, victim_type=victim_type, start=start, end=end, comp_mode=comp_mode, data_mode=data_mode)

    if further_descr:
        filename_batches = os.path.join(dir_path, "Data/", f"{subfolder}/", f"{filename_identifier}_{further_descr}_batches_attacked.pickle")
        filename_doc_chunkings = os.path.join(dir_path, "Data/", f"{subfolder}/", f"{filename_identifier}_{further_descr}_doc_chunkings_attacked.pickle")
        filename_documents = os.path.join(dir_path, "Data/", f"{subfolder}/", f"{filename_identifier}_{further_descr}_documents_attacked.pickle")
    else:
        filename_batches = os.path.join(dir_path, "Data/", f"{subfolder}/", f"{filename_identifier}_batches_attacked.pickle")
        filename_doc_chunkings = os.path.join(dir_path, "Data/", f"{subfolder}/", f"{filename_identifier}_doc_chunkings_attacked.pickle")
        filename_documents = os.path.join(dir_path, "Data/", f"{subfolder}/", f"{filename_identifier}_documents_attacked.pickle")

    with open(filename_batches, "wb") as outfile_batches:
        pickle.dump(attacked_batches, outfile_batches, pickle.HIGHEST_PROTOCOL)
        
    with open(filename_doc_chunkings, "wb") as outfile_doc_chunkings:
        pickle.dump(attacked_doc_chunkings, outfile_doc_chunkings, pickle.HIGHEST_PROTOCOL)

    with open(filename_documents, "wb") as outfile_documents:
        pickle.dump(attacked_documents, outfile_documents, pickle.HIGHEST_PROTOCOL)

def __fill_comp_predictions__(max_len, prediction):
    FILLER = 0
    prediction_ = []
    none_list = [FILLER for i in range(max_len)]
    for i in range(len(prediction)):
        if not prediction[i]:
            prediction_.append(none_list)

        elif len(prediction[i]) < max_len:
            diff = max_len - len(prediction[i])
            prediction[i].extend([FILLER for i in range(diff)])
            prediction_.append(prediction[i])

        elif len(prediction[i]) == max_len:
            prediction_.append(prediction[i])
    return prediction_



    