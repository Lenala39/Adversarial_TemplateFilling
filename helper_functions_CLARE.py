import functools
from tracemalloc import start
import spacy_alignments
import tensorflow as tf
from helper_functions import pretty_print_difference

def update_entities_CLARE(sentence_obj, original_tokens, new_sentence_tokens, idx, attack_type):

    #original2new, new2original = spacy_alignments.get_alignments(original_tokens, new_sentence_tokens)
    attacked_entities, following_entities = find_attacked_and_following_entities(idx, sentence_obj.get_entities())

    length_diff = len(new_sentence_tokens) - len(original_tokens)


    for attacked_ent, start_end_pos in attacked_entities:
        new_entity_tokens = new_sentence_tokens[start_end_pos[0]: start_end_pos[1] + 1 + length_diff]
        
        attacked_ent.set_tokens(new_entity_tokens)

        old_start_pos = attacked_ent.get_start_pos()
        old_end_pos = attacked_ent.get_end_pos()
        
        if old_start_pos > idx:
            attacked_ent.set_start_pos(old_start_pos + length_diff)
        
        attacked_ent.set_end_pos(old_end_pos + length_diff)
        assert new_sentence_tokens[attacked_ent.get_start_pos():attacked_ent.get_end_pos()+1] == new_entity_tokens, "new_sentence_tokens with indices dont match to new tokens"
    

    for ent, _ in following_entities:
        
        old_start_pos = ent.get_start_pos()
        old_end_pos = ent.get_end_pos()
        old_tokens = ent.get_tokens()
        try:
            assert original_tokens[old_start_pos:old_end_pos+1] == old_tokens, "original tokens dont match to indices in original sentence"
        except AssertionError as e:
            if len(old_tokens) == 0:
                ent.set_start_pos(ent.get_end_pos()+1)
                old_start_pos = ent.get_start_pos()
                old_end_pos = ent.get_end_pos()
        try:        
            assert original_tokens[ent.get_start_pos():old_end_pos+1] == old_tokens, "original tokens dont match to indices in original sentence"
        except AssertionError as e:
            print(e)
        if old_start_pos > idx:
        
            ent.set_start_pos(old_start_pos + length_diff)

        ent.set_end_pos(old_end_pos +length_diff)

        if attack_type == "merge" and old_start_pos == idx + 1 and length_diff < 0: # merge also affects position after it
            # do not need to change start pos BUT since word from entity was merged "away", need to change tokens
            new_entity_tokens = new_sentence_tokens[ent.get_start_pos():ent.get_end_pos() + 1]#+ 2 + length_diff]
            ent.set_tokens(new_entity_tokens)
        elif attack_type == "merge" and old_start_pos == idx + 1 and length_diff == 0:
            if ent.get_start_pos() < ent.get_end_pos():
                ent.set_start_pos(ent.get_start_pos()+1)
            new_entity_tokens = new_sentence_tokens[ent.get_start_pos():ent.get_end_pos() + 1]
            ent.set_tokens(new_entity_tokens)

        elif attack_type == "merge" and old_start_pos == idx + 1 and length_diff > 0:
            ent.set_start_pos(ent.get_start_pos()+1)
            new_entity_tokens = new_sentence_tokens[ent.get_start_pos():ent.get_end_pos() + 1]
            ent.set_tokens(new_entity_tokens)
        
        elif attack_type == "merge" and ent.get_start_pos() == ent.get_end_pos() and len(ent.get_tokens()) == 0:
            ent.set_start_pos(ent.get_start_pos()+1) 
        
        try:
            assert new_sentence_tokens[ent.get_start_pos():ent.get_end_pos()+1] == ent.get_tokens(), "sentence tokens + indices dont match to entity tokens"
        except AssertionError:
            #print(new_sentence_tokens[ent.get_start_pos():ent.get_end_pos()+1])
            #print(ent.get_tokens())
            #print(attack_type)
            pass

def find_attacked_and_following_entities(idx, entities):
    
    
    attacked_entities = []
    following_entities = []
    
    for entity in entities:
        # entity was found -> word inside entity
        start_pos = entity.get_start_pos()
        end_pos = entity.get_end_pos()
        if start_pos <= idx and end_pos >= idx:
            attacked_entities.append((entity, (start_pos, end_pos)))
        elif start_pos >= idx:
            following_entities.append((entity, (start_pos, end_pos)))
    
    return attacked_entities, following_entities

def print_attack_logs(attack_logs, s_index, new_sentence, old_sentence, changed_sentences):
    print("----------------------------------")
    print(f"sentence number: {s_index}")
    if len(changed_sentences) > 0:
        print(f"Entities were changed with divergence {attack_logs[-1]['divergence']} after {attack_logs[-1]['num_changed']} changes using {attack_logs[-1]['num_queries']} queries!")
                
        for i in range(len(attack_logs)):
            if attack_logs[i]["attack_type"] == "insert":
                print(f"Attack: Inserted {attack_logs[i]['synonym_tokens']} after {attack_logs[i]['original_tokens']}")
            elif attack_logs[i]["attack_type"] == "replace":
                print(f"Attack: Replaced {attack_logs[i]['original_tokens']} with {attack_logs[i]['synonym_tokens']}")
            elif attack_logs[i]["attack_type"] == "merge":
                print(f"Attack: Merged {attack_logs[i]['original_tokens']} with following token to {attack_logs[i]['synonym_tokens']} ")

        pretty_print_difference(old_sentence, new_sentence)

    else:
        print("No entities were changed, attack failed!")

def print_attack_logs_comp(attack_logs, s_index, new_sentence, old_sentence):
    print("----------------------------------")
    print(f"sentence number: {s_index}")
    if len(attack_logs) > 0 and attack_logs[-1]["attack_successful"]:
        print(f"Compatibilities were changed after {attack_logs[-1]['num_changed']} changes using {attack_logs[-1]['num_queries']} queries!")   
        for i in range(len(attack_logs)):
            
            if attack_logs[i]["attack_type"] == "insert":
                print(f"Attack: Inserted {attack_logs[i]['synonym_tokens']} after {attack_logs[i]['original_tokens']}")
            elif attack_logs[i]["attack_type"] == "replace":
                print(f"Attack: Replaced {attack_logs[i]['original_tokens']} with {attack_logs[i]['synonym_tokens']}")
            elif attack_logs[i]["attack_type"] == "merge":
                print(f"Attack: Merged {attack_logs[i]['original_tokens']} with following token to {attack_logs[i]['synonym_tokens']} ")

        pretty_print_difference(old_sentence, new_sentence)

    elif len(attack_logs) > 0 and not attack_logs[-1]["attack_successful"]:
        print(f"No compatibilities were changed, attack failed BUT attacks {len(attack_logs)} were performed!")
        
        if attack_logs[0]["divergence"] and attack_logs[-1]["divergence"]:
            print("divergences: ")
            for i in range(len(attack_logs)):
                try:
                    end_token = "<" if attack_logs[i+1]["divergence"] > attack_logs[i]["divergence"] else ">"
                except IndexError:
                    end_token = ""
                except TypeError:
                    end_token = ""
            
                print(format(attack_logs[i]["divergence"], ".20f"), end=f" {end_token} ")
    else:
        print("No compatibilities were changed, attack failed!")



if __name__ == "__main__":
    import os 
    disease_str = "gl"
    dir_path = os.path.dirname(os.path.realpath(__file__)) 
    filename = os.path.join(dir_path, "attack_logs/" f"CLARE_{disease_str}_results.json")
    print(filename)
    exit()
    
    # import tensorflow as tf
    import numpy as np
    #split_prediction_into_sentences(prediction=None, chunk=None)
    a=np.random.randint(5,size=(1,10))
    b=np.random.randint(5,size=(1,10))

    a1=tf.convert_to_tensor(a,dtype=tf.float32)
    b1=tf.convert_to_tensor(b,dtype=tf.float32)

    result=tf.add(a1,-b1)
