from helper_functions import pretty_print_difference
import re

import functools
from pymedtermino_master.snomedct import *
from pymedtermino_master.icd10 import ICD10

from pymedtermino_master.umls import *

# mysql -uroot --password="lenadb123" -h 127.0.0.1 -P3306
connect_to_umls_db(host='127.0.0.1', user="root", password="lenadb123", db_name="umls", encoding="utf8_unicode_ci")

extract_letter_and_code = re.compile("(\w)(\d{1,2})(\.(\d{1,2}))?")

@functools.lru_cache(maxsize=128)
def search_umls(search_term):
    results = UMLS_CUI.search(search_term)
    synonyms = []
    for res in results:
        synonyms.extend(res.terms)
    # sub stuff in brackets for nothing
    synonyms = [re.sub("(\([\w|\s]*\))", "", elem).strip() for elem in synonyms if not re.sub("(\([\w|\s]*\))", "", elem).lower().strip() == search_term.lower()]
    return list(set(synonyms))


@functools.lru_cache(maxsize=128)
def find_closest_match(search_term, nlp, search_db="umls", chemical=False):
    #print("Searching for all candidates")
    if search_db.lower() == "umls":
        candidate_pool = search_umls(search_term)

    if not candidate_pool:
        #print(f"No candidates found for {search_term}")
        return None

    # make term a spacy "document" to be able to compare it
    doc_search_term = nlp(search_term)
    
    # find the most similar one from the candidates (by name)
    max_similarity = 0
    max_match = None 
        
    for elem in candidate_pool:
        # find most similar candidate by name
        doc_elem = nlp(elem)
        similarity = doc_search_term.similarity(doc_elem)
        # print(f"{search_term} vs. {elem} = {similarity}")
        # similarity: 1 = identical, 0 = not at all similar
        if similarity < 1 and similarity >= 0.7 and similarity > max_similarity:
            max_similarity = similarity
            max_match = elem
    
    
    return max_match


def __find_synonym__(spacy_entity, sent_obj, spacy2data, nlp, find_entity=True):
    most_similar = None

    original_start_of_entity = spacy2data[spacy_entity.start][0]
    original_end_of_entity = spacy2data[spacy_entity.start][-1]
    if find_entity:
        modified_entity, exact_match = find_modified_entity(original_start_of_entity, original_end_of_entity, spacy_entity.text.strip(), sent_obj)

        if modified_entity and exact_match:
            search_term = spacy_entity.text.strip()
            
        elif modified_entity and not exact_match:
            search_term = modified_entity.get_string()
        # no entity found
        else:
            return None, None, None
    else:
        search_term = spacy_entity.text.strip()
        modified_entity = None 
        exact_match = None
        
    # NE: replace using PyMedTermino (chemicals / diseases only) -> replace most similar
    if spacy_entity.label_ == "DISEASE":
        most_similar = find_closest_match(search_term, nlp, search_db="umls", chemical=False)
    # NE: replace using PyMedTermino (chemicals / diseases only) -> replace most similar
    elif spacy_entity.label_ == "CHEMICAL":
        most_similar = find_closest_match(search_term, nlp, search_db="umls", chemical=True)
    
    return most_similar, modified_entity, exact_match


def find_modified_entity(original_start_of_entity, original_end_of_entity, spacy_string, sent_obj):
    # check if there was an entity with that start pos -> update tokens, indices
    for e in sent_obj.get_entities():
        if e.get_start_pos() == original_start_of_entity and e.get_end_pos() == original_end_of_entity:
            return e, True
    
    for e in sent_obj.get_entities():
        if e.get_start_pos() <= original_start_of_entity + 5 and e.get_start_pos() >= original_start_of_entity - 5:
            if spacy_string in e.get_string(): 
                return e, False 
    
    return None, False


@functools.lru_cache(maxsize=128)
def search_icd10(search_term):
    search_results = ICD10.search(search_term)

    if len(search_results) == 0:
        return None
    # standardize search term
    search_term = search_term.lower()

    selected_code = None
    # search the exact match to find the Code
    for elem in search_results:
        if search_term.lower() == elem.term.lower():

            if not "-" in elem.code:
                selected_code = elem.code
                break 
    
    # iterate results: check that string is close (elem.term) AND that elem.code start with the same main number
    # make candidate pool based on the Code that was extracted (only same code e.g. H40 -> H40.1, ... BUT NOT H25)
    candidate_pool = []
    if selected_code:
        result = extract_letter_and_code.search(selected_code).groups()
        letter_id = result[0]
        primary_class = result[1]
        secondary_class = result[3]
        print("ICD10: Checking that all elems have similar code")
        for elem in search_results:
            result_elem = extract_letter_and_code.search(elem.code).groups()

            letter_id_temp = result_elem[0]
            primary_class_temp = result_elem[1]
            secondary_class_temp = result_elem[3]

            if letter_id == letter_id_temp and primary_class_temp == primary_class:
                candidate_pool.append(elem.term.lower())
    
    # all entries are in the candidate pool
    else:
        candidate_pool = [elem.term.lower() for elem in search_results if not elem.term.lower == search_term.lower()]
    
    return candidate_pool


@functools.lru_cache(maxsize=128)
def search_snomed(search_term, chemical):
    search_results = SNOMEDCT.search(search_term)
    if len(search_results) == 0:
        return None

    exact_match = None
    for elem in search_results:
        cleaned_term = re.sub("(\([\w|\s]*\))", "", elem.term.lower()).strip()
        if cleaned_term == search_term.lower():
            exact_match = elem
            break
    
    synonyms = []
    if exact_match:
        synonyms = exact_match.terms
        # print(synonyms)
    # synonyms are found for the search_term
    if len(synonyms) > 0:
        candidate_pool = [re.sub("(\([\w|\s]*\))", "", elem.term.lower()).strip() for elem in search_results]
    # no synonyms found or no exact match found
    else:
        if chemical:
            candidate_pool = [re.sub("(\([\w|\s]*\))", "", elem.term.lower()).strip() for elem in search_results if not elem.is_a("disease")]
        else:
            candidate_pool = [re.sub("(\([\w|\s]*\))", "", elem.term.lower()).strip() for elem in search_results if elem.is_a("disease")]
            #if not elem.term.lower == search_term.lower() and 
    
    return candidate_pool




def update_affected_entity(e, tokenized_match, original_start_of_entity):
    new_end_index = original_start_of_entity + len(tokenized_match) - 1
    old_end_index = e.get_end_pos() # is the same as original_end_of_entity
    e.set_end_pos(new_end_index)
    e.set_tokens(tokenized_match)
    # positive if the new one is longer (latanoprost -> topical latanoprost)
    length_diff_between_orig_and_synonym = new_end_index - old_end_index
    return length_diff_between_orig_and_synonym


def update_spanning_entity(spanning_entity, tokenized_match, original_start_of_entity, original_end_of_entity, elem):
    if elem.text in spanning_entity.get_string():
        spanning_start = spanning_entity.get_start_pos()
        start_offset = original_start_of_entity - spanning_start

        old_tokens = spanning_entity.get_tokens()
        # number of tokens in the originial entity e.g. ['la', '##tan', '##op', '##ros', '##t'] -> 5
        original_num_of_tokens = original_end_of_entity - original_start_of_entity + 1
        before_entity = old_tokens[0:start_offset]
        after_entity = old_tokens[start_offset+original_num_of_tokens:]
        
        new_tokens = before_entity + tokenized_match + after_entity
        spanning_entity.set_tokens(new_tokens)

def update_spanning_before(spanning_entity, tokenized_match, original_start_of_entity, original_end_of_entity, elem):
    if isinstance(elem, list):
        elem = " ".join(elem).replace(" ##", "")
    else:
        elem = elem.text
    # only update if entity is actually in the spanning entity
    if elem in spanning_entity.get_string():
        # end pos
        spanning_start = spanning_entity.get_start_pos()
        start_offset = original_start_of_entity - spanning_start

        old_tokens = spanning_entity.get_tokens()
        # number of tokens in the originial entity e.g. ['la', '##tan', '##op', '##ros', '##t'] -> 5
        original_num_of_tokens = original_end_of_entity - original_start_of_entity + 1
        before_entity = old_tokens[0:start_offset]
        after_entity = old_tokens[start_offset+original_num_of_tokens:]
        
        new_tokens = before_entity + tokenized_match + after_entity
        spanning_entity.set_tokens(new_tokens)

def print_attack_logs(attack_logs, s_index, new_sentence, old_sentence, changed_sentences):
    print("----------------------------------")
    print(f"sentence number: {s_index}")
    if len(changed_sentences) > 0:
        print(f"Entities were changed after {attack_logs[-1]['num_changed']} changes using {attack_logs[-1]['num_queries']} queries!")
                
        for i in range(len(attack_logs)):
            
            print(f"Attack: Replaced {attack_logs[i]['original_tokens']} ({attack_logs[i]['type']}) with {attack_logs[i]['synonym_tokens']}")
            
        pretty_print_difference(old_sentence, new_sentence)

    else:
        print("No entities were changed, attack failed!")

def print_attack_logs_comp(attack_logs, s_index, new_sentence, old_sentence):
    print("----------------------------------")
    print(f"sentence number: {s_index}")
    if len(attack_logs) > 0 and attack_logs[-1]["attack_successful"]:
        print(f"Compatibilities were changed after {attack_logs[-1]['num_changed']} changes using {attack_logs[-1]['num_queries']} queries!")
        print(f"Final divergence was {attack_logs[-1]['divergence']:.20f}")   
        for i in range(len(attack_logs)):
            
            print(f"Attack: Replaced {attack_logs[i]['original_tokens']} ({attack_logs[i]['type']}) with {attack_logs[i]['synonym_tokens']}")
            
        pretty_print_difference(old_sentence, new_sentence)

    elif len(attack_logs) > 0 and not attack_logs[-1]["attack_successful"] and attack_logs[-1]["token_index"]: # if no attacks computed, then everythin except s_index and a_success is NOne
        print(f"No compatibilities were changed, divergence was {attack_logs[-1]['divergence']:.20f}, BUT {len(attack_logs)} attacks were performed!")
    
    else:
        print("No compatibilities were changed, attack failed!")