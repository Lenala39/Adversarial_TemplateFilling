from urllib.parse import _NetlocResultMixinStr
from torch import sigmoid
import SlotFillingCompModule
from SlotFillingCompModule import SlotFillingCompModule, create_batches_from_chunking, prepare_module

from helper_functions import __compute_divergence__, parse_cli_arguments
from helper_functions import *
from helper_functions_Araujo import __find_synonym__
from helper_functions_Araujo import *


import spacy_alignments

import os 
from tqdm import tqdm

import spacy
import scispacy

from official.nlp.bert import tokenization
import tensorflow_hub as hub

from DocumentEncoder import DocumentEncoder

disease_str, victim_type, start, end, comp_mode, data_mode = parse_cli_arguments()

print("Loading spaCy model")
# TODO: NER as only pipeline element: https://spacy.io/api/entityrecognizergithub

nlp = spacy.load("en_ner_bc5cdr_md")#, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "textcat", "ner"], 
                                    #exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "textcat", "ner"])
print("Done loading spaCy model")

if disease_str == "gl":
    model_prefix = 'glaucoma'
else: 
    model_prefix = "dm2"

dir_path = os.path.dirname(os.path.realpath(__file__))
ANNOTATED_DATA_DIR = os.path.join(dir_path, "Data", "AnnotatedCorpus")
MODEL_CHECKPOINT_FOLDER = os.path.join(dir_path, "Models", disease_str)

MIN_SLOT_FREQ = 20
MAX_CHUNK_SIZE = 400
BERT_MODEL_DIM = 768    


bert_model_name = "https://tfhub.dev/google/experts/bert/pubmed/2"
FullTokenizer = tokenization.FullTokenizer
bert_layer = hub.KerasLayer(bert_model_name, trainable=False)
# The vocab file of bert for tokenizer
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
bert_tokenizer = FullTokenizer(vocab_file)

# construct the module for data and victim model
slot_filling_module = prepare_module(model_prefix=model_prefix, load_from_pickle=True, train=False, model="Araujo", 
                                        comp_mode=comp_mode, load_augmented=False, victim_type=victim_type,
                                        load_slot_indices=True, model_name=None, load_weights=True)

victim = slot_filling_module.slot_filling_model

# create document encoder
document_encoder = DocumentEncoder(bert_tokenizer, slot_filling_module.slot_indices, 'no_slot')


def apply_araujo(document, doc_chunking, batch, chunk, sent_obj, comp_mode, victim_type, original_prediction, original_prediction_split, sigmoid=False):
    # perform NER (spacy)
    # sentence_string = " ".join(sent_obj.get_tokens()).replace(" ##", "")
    # TODO: use nlp.pipe(texts) for faster processing of batches of text
    
    # get sentence tokens and make into string
    sentence_string = " ".join(sent_obj.get_tokens()).replace(" ##", "")
    
    doc = nlp(sentence_string)
    spacy_tokens = [str(doc[i]) for i in range(len(doc))]
    
    original_tokens = sent_obj.get_tokens().copy()

    # compute alignment for mapping back and forth
    _, spacy2data = spacy_alignments.get_alignments(sent_obj.get_tokens(), spacy_tokens)
    
    num_changed = 0
    attack_logs = []

    for spacy_entity in doc.ents:
        original_token_start = spacy2data[spacy_entity.start][0]
        most_similar, modified_entity, exact_match = __find_synonym__(spacy_entity=spacy_entity, sent_obj=sent_obj, spacy2data=spacy2data, nlp=nlp)

        # synoynm was found for term
        if most_similar:            
            tokenized_match = bert_tokenizer.tokenize(most_similar)

            __update_entities__(sent_obj, modified_entity, tokenized_match, spacy_entity)

            num_changed += 1
            # recompute the alignment for matching
            _, spacy2data = spacy_alignments.get_alignments(sent_obj.get_tokens(), spacy_tokens)
            
            temp = {
                "sentence_index": sent_obj.get_index(), 
                "token_index": original_token_start,
                "type": spacy_entity.label_,  
                "original_tokens": str(spacy_entity.text), 
                "synonym_tokens": " ".join(tokenized_match).replace(" ##", ""),
                "synonym_prob": None,
                "semantic_sim": None,
                "num_changes_in_pred": None, 
                "num_changed": num_changed,
                "num_queries": num_changed,
                "changed_sentences": None,
                "num_entities_changed": None,  
                "comp_mode": comp_mode,
                "attack_successful": None, 

            }
            attack_logs.append(temp)
    
    sentences_with_changed_entities = []
    
    if victim_type == "extraction":
        attack_logs, sentences_with_changed_entities, sent_obj, attack_successfull = __check_entities_changed_extraction__(attack_logs=attack_logs, doc_chunking=doc_chunking, 
                                                                                                    chunk=chunk, sent_obj=sent_obj, original_prediction_split=original_prediction_split,
                                                                                                    original_tokens=original_tokens, original_prediction=original_prediction)

    elif victim_type == "comp":
        attack_logs, sent_obj, attack_successfull = __check_entities_changed_comp__(attack_logs=attack_logs, doc_chunking=doc_chunking, document=document, 
                                                                                        chunk=chunk,sent_obj=sent_obj, victim_type=victim_type, 
                                                                                        original_prediction=original_prediction, original_prediction_split=original_prediction_split)
                                                                                        

    return sent_obj, attack_logs, sentences_with_changed_entities, attack_successfull

def __update_entities__(sent_obj, modified_entity, tokenized_match, elem):
    original_start_of_entity = modified_entity.get_start_pos()
    original_end_of_entity = modified_entity.get_end_pos()

    length_diff_between_orig_and_synonym = update_affected_entity(modified_entity, tokenized_match, original_start_of_entity)
    
    # update sentence -> do first since debugging entity updates is easier then
    before_entity = sent_obj.get_tokens()[:original_start_of_entity]
    after_entity = sent_obj.get_tokens()[original_end_of_entity+1:]

    new_tokens = before_entity + tokenized_match + after_entity
    old_length = len(sent_obj.get_tokens())
    sent_obj.set_tokens(new_tokens)
    assert len(sent_obj.get_tokens()) == old_length + length_diff_between_orig_and_synonym

    
    # update all entities following the modified one -> index/pos has to be shifted
    for e in sent_obj.get_entities():

        # skip the modified entity itself
        if not e == modified_entity:

            # entity is after the modified entity in the sentence
            if e.get_start_pos() >= original_end_of_entity:
                old_start = e.get_start_pos()
                e.set_start_pos(old_start + length_diff_between_orig_and_synonym)

                old_end = e.get_end_pos()
                e.set_end_pos(old_end + length_diff_between_orig_and_synonym)
                # TODO: still get error sometimes
                try:
                    assert e.get_tokens() == sent_obj.get_tokens()[e.get_start_pos():e.get_end_pos()+1]
                except AssertionError:
                    print(e.get_tokens())
                    print(sent_obj.get_tokens()[e.get_start_pos():e.get_end_pos()+1])

            # entity starts before but ends within 
            elif e.get_start_pos() <= original_start_of_entity and e.get_end_pos() <= original_end_of_entity and e.get_end_pos() >= original_start_of_entity:
                update_spanning_entity(e, tokenized_match, original_start_of_entity, original_end_of_entity, elem)
                old_end = e.get_end_pos()
                e.set_end_pos(old_end + length_diff_between_orig_and_synonym)
            
                try:
                    assert e.get_tokens() == sent_obj.get_tokens()[e.get_start_pos():e.get_end_pos()+1]
                except AssertionError:
                    print(e.get_tokens())
                    print(sent_obj.get_tokens()[e.get_start_pos():e.get_end_pos()+1])

            # entity starts within but ends after 
            elif e.get_start_pos() >= original_start_of_entity and e.get_end_pos() >= original_end_of_entity and e.get_start_pos() <= original_end_of_entity:
                update_spanning_entity(e, tokenized_match, original_start_of_entity, original_end_of_entity, elem)
                old_end = e.get_end_pos()
                e.set_end_pos(old_end + length_diff_between_orig_and_synonym)
            
                try:
                    assert e.get_tokens() == sent_obj.get_tokens()[e.get_start_pos():e.get_end_pos()+1]
                except AssertionError:
                    print(e.get_tokens())
                    print(sent_obj.get_tokens()[e.get_start_pos():e.get_end_pos()+1])
            
            # entity starts within and ends within
            elif e.get_start_pos() >= original_start_of_entity and e.get_start_pos() <= original_end_of_entity and e.get_end_pos() <= original_end_of_entity and e.get_end_pos() >= original_start_of_entity:
                update_spanning_entity(e, tokenized_match, original_start_of_entity, original_end_of_entity, elem)
                old_end = e.get_end_pos()
                e.set_end_pos(old_end + length_diff_between_orig_and_synonym)
            
                try:
                    assert e.get_tokens() == sent_obj.get_tokens()[e.get_start_pos():e.get_end_pos()+1]
                except AssertionError:
                    print(e.get_tokens())
                    print(sent_obj.get_tokens()[e.get_start_pos():e.get_end_pos()+1])

            # only the end_pos is after the modified entity, entity starts before the subsituted entity
            elif e.get_start_pos() < original_start_of_entity and e.get_end_pos() >= original_end_of_entity and not e == modified_entity:
                update_spanning_entity(e, tokenized_match, original_start_of_entity, original_end_of_entity, elem)
                old_end = e.get_end_pos()
                e.set_end_pos(old_end + length_diff_between_orig_and_synonym)
                
                try:
                    assert e.get_tokens() == sent_obj.get_tokens()[e.get_start_pos():e.get_end_pos()+1]
                except AssertionError:
                    print(e.get_tokens())
                    print(sent_obj.get_tokens()[e.get_start_pos():e.get_end_pos()+1])


def __compute_model_pred__(document, batch, doc_chunking,victim_type, sigmoid, comp_mode):
    
    if victim_type == "extraction":
        prediction = slot_filling_module.get_start_end_positions([batch]) 
        prediction_split = split_prediction_into_sentences(prediction, doc_chunking)
    else:
        comp_scores, comp_scores_no_sigmoid = slot_filling_module.compute_entity_comp_for_attacking(document, doc_chunking, [batch], sigmoid=sigmoid)
        prediction = comp_scores 
        prediction_split = comp_scores_no_sigmoid

    return prediction, prediction_split


def __check_entities_changed_comp__(attack_logs, doc_chunking, document, chunk, sent_obj, original_prediction, original_prediction_split, victim_type, sigmoid=False):

    if len(attack_logs) == 0:
        temp = {
            "sentence_index": sent_obj.get_index(), 
            "token_index": None,
            "type": None,
            "original_tokens": None,
            "synonym_tokens": None,
            "synonym_prob": None,
            "semantic_sim": None,
            "num_changes_in_pred": None,
            "num_changed": None,
            "num_queries": None,
            "changed_sentences": None,
            "num_entities_changed": None, 
            "comp_mode": None,
            "attack_successful": False
        }
        attack_logs.append(temp)
        return attack_logs, sent_obj, False 
    
    # check if prediction was changed 
    # set sentence in chunk
    original_chunks = doc_chunking.get_chunks()
    original_chunks = [c.__deepcopy__(memo={}) for c in original_chunks]        

    chunk.set_sentence_by_index(sent_obj.get_index(), sent_obj)
    
    # re-compute doc_chunking from chunks
    chunk = doc_chunking.set_from_chunks(chunk, sent_obj.get_index()) 
    # recompute sentence offsets in chunk -> sentence lenght might have changed
    
    # compute batch from doc_chunking again
    new_batch = create_batches_from_chunking([doc_chunking], document_encoder, slot_filling_module.slot_indices)
    new_prediction, new_prediction_split = __compute_model_pred__(document=document, batch=new_batch[0], doc_chunking=doc_chunking, victim_type=victim_type, sigmoid=sigmoid, comp_mode=comp_mode)

    attack_successful, num_changes_in_pred, divergence = compute_compatibility_changed(original_pred=original_prediction, new_pred=new_prediction, 
                                                                                            original_prediction_split=original_prediction_split, 
                                                                                            new_prediction_split=new_prediction_split, 
                                                                                            victim_type=victim_type, mode=comp_mode)

    
    attack_logs[-1]["num_changes_in_pred"] = num_changes_in_pred
    attack_logs[-1]["attack_successful"] = attack_successful
    attack_logs[-1]["divergence"] = divergence

    doc_chunking.set_all_chunks(original_chunks)

    return attack_logs, sent_obj, True


def __check_entities_changed_extraction__(attack_logs, doc_chunking, chunk, sent_obj, original_prediction_split, original_prediction, original_tokens):
    
    sentences_with_changed_entities = []

    if len(attack_logs) == 0:
        temp = {
            "sentence_index": sent_obj.get_index(), 
            "token_index": None,
            "type": None,
            "original_tokens": None,
            "synonym_tokens": None,
            "synonym_prob": None,
            "semantic_sim": None,
            "num_changes_in_pred": None,
            "num_changed": None,
            "num_queries": None,
            "changed_sentences": None,
            "num_entities_changed": None, 
            "comp_mode": None,
            "attack_successful": False 

        }
        attack_logs.append(temp)
        return attack_logs, sentences_with_changed_entities, sent_obj, False 
    
    # check if prediction was changed 
    # set sentence in chunk
    original_chunks = doc_chunking.get_chunks()
    original_chunks = [c.__deepcopy__(memo={}) for c in original_chunks]        

    chunk.set_sentence_by_index(sent_obj.get_index(), sent_obj)
    
    # re-compute doc_chunking from chunks
    chunk = doc_chunking.set_from_chunks(chunk, sent_obj.get_index()) 
    # recompute sentence offsets in chunk -> sentence lenght might have changed
    
    # compute batch from doc_chunking again
    new_batch = create_batches_from_chunking([doc_chunking], document_encoder, slot_filling_module.slot_indices)
    new_prediction = slot_filling_module.get_start_end_positions(new_batch) 

    new_prediction_split = split_prediction_into_sentences(new_prediction, doc_chunking)
    sentences_with_changed_entities, num_entities_changed = compute_entity_changed(original_prediction_split, new_prediction_split, 
                                                                original_tokens=original_tokens, doc_chunking=doc_chunking, sentence_object=sent_obj, 
                                                                slot_indices=slot_filling_module.slot_indices)

    attack_successful = True if len(sentences_with_changed_entities) > 0 else False
    divergence = float(__compute_divergence__(original_prediction=original_prediction, new_prediction=new_prediction, victim_type=victim_type))
    
    # set last temp["changed_sentences"] and temp["attack_successful"] (attack_logs[-1])
    attack_logs[-1]["changed_sentences"] = sentences_with_changed_entities # check why list always empty!!
    attack_logs[-1]["attack_successful"] = attack_successful
    attack_logs[-1]["num_entites_changed"] = num_entities_changed
    attack_logs[-1]["divergence"] = divergence 

    doc_chunking.set_all_chunks(original_chunks)

    return attack_logs, sentences_with_changed_entities, sent_obj, True
        

def run_attack(slot_filling_module, comp_mode, data_mode, start, end, victim_type, model="Araujo"):
    output = []
    attacked_doc_chunkings = []
    attacked_documents = []
    attacked_batches = []
    attacked_sentences = []

    documents, document_chunkings, batches = get_data_depending_on_split(slot_filling_module, data_mode=data_mode)
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

        original_prediction, original_prediction_split = __compute_model_pred__(document=document, doc_chunking=doc_chunking, batch=batch, victim_type=victim_type, sigmoid=sigmoid, comp_mode=comp_mode)

        # iterate over chunks
        for c_index in range(len(doc_chunking.get_chunks())):
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


                ### ARAUJO specific 
                # attack
                new_sentence, attack_logs, changed_sentences, attack_successful = apply_araujo(document=document, doc_chunking=doc_chunking, batch=batch, chunk=chunk, sent_obj=sentence, victim_type=victim_type, 
                                                                                original_prediction=original_prediction, original_prediction_split=original_prediction_split,
                                                                                comp_mode=comp_mode) 
                
            
                # print logs
                if victim_type == "extraction":
                    print_attack_logs(attack_logs, s_index, new_sentence.get_tokens(), old_tokens, changed_sentences)
                else:
                    print_attack_logs_comp(attack_logs, s_index, new_sentence.get_tokens(), old_tokens)
                    
                # if the attack was not successful -> reset entities to original
                if not attack_successful:
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
    
    if not start:
        start = 0
    if not end and data_mode == "train":
        end = len(slot_filling_module.documents_train)-1
    elif not end and data_mode == "test":
        end = len(slot_filling_module.documents_test)-1
    
    print(victim_type)
    run_attack(slot_filling_module, model="Araujo", victim_type=victim_type, comp_mode=comp_mode, data_mode=data_mode, start=start, end=end)