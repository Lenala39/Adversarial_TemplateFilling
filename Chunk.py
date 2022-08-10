import numpy as np
from copy import deepcopy


class Chunk:
    def __init__(self, sentences, chunk_index):
        self._sentences = list(sentences)
        
        # chunk index within DocumentChunking
        self._chunk_index = chunk_index
        
        # token offsets of sentences in chunk
        # key: sentence index
        # value: token offset of sentence in chunk
        self._sentence_offsets = dict()
                
        # set sentence offsets
        current_sentence_offset = 1 # [CLS] token is first token in each chunk
        
        for sentence in self.get_sentences():
            # get index of sentence
            sentence_index = sentence.get_index()
            assert sentence_index is not None, "Sentence index is None in Chunk __init__"
            
            # srt in chunk offset of sentence
            self._sentence_offsets[sentence_index] = current_sentence_offset
            current_sentence_offset += sentence.get_num_tokens() + 1 # + 1 because of [SEP] token
            
    def update_sentence_offsets(self):
        # recompute sentence offsets -> sentence length might change for modification
        self._sentence_offsets = dict()
         # set sentence offsets
        current_sentence_offset = 1 # [CLS] token is first token in each chunk
        
        for sentence in self.get_sentences():
            # get index of sentence
            sentence_index = sentence.get_index()
            assert sentence_index is not None, "Sentence index is None in Chunk.update_sentence_offset"
            
            # srt in chunk offset of sentence
            self._sentence_offsets[sentence_index] = current_sentence_offset
            current_sentence_offset += sentence.get_num_tokens() + 1 # + 1 because of [SEP] token
    
    def set_sentences(self, new_sentences):
        self._sentences = []
        for s in new_sentences:
            self._sentences.append(s)
        self.update_sentence_offsets()


    def get_sentences(self):
        return self._sentences
    
    
    def get_num_sentences(self):
        return len(self._sentences)
    
    
    def set_chunk_index(self, chunk_index):
        self._chunk_index = chunk_index
        
        
    def get_chunk_index(self):
        return self._chunk_index
    
    
    def get_sentence_indices(self):
        return {sentence.get_index() for sentence in self.get_sentences()}
    
    
    def get_sentence_by_index(self, sentence_index):
        if sentence_index not in self._sentence_offsets:
            # raise IndexError('Invalid sentence index')
            return None
            
        for sentence in self.get_sentences():
            if sentence.get_index() == sentence_index:
                return sentence
            
        # sentence was not found
        raise Exception('Sentence not found in chunk')
    
    
    def get_sentence_offset(self, sentence_index):
        if sentence_index not in self._sentence_offsets:
            # raise IndexError('Invalid sentence index')# 
            return None
            
        return self._sentence_offsets[sentence_index]
    
    
    def extract_sentence_subarray(self, chunk_array, sentence_index):
        # estimate sentence boundaries
        sentence = self.get_sentence_by_index(sentence_index)
        sentence_start_offset = self.get_sentence_offset(sentence_index)
        sentence_end_offset = sentence_start_offset + sentence.get_num_tokens()

        # return numpy subarray of chunk
        return chunk_array[sentence_start_offset:sentence_end_offset]
    
    
    def set_entity_chunk_indices(self):
        chunk_index = self
            
        
    def set_sentence_by_index(self, sentence_index, new_sentence):
        
        list_index = self.get_mapping_sentence_index_to_list_index(sentence_index)

        self._sentences[list_index] = new_sentence

    def get_mapping_sentence_index_to_list_index(self, sentence_index):
        if sentence_index not in self._sentence_offsets:
            raise IndexError('Invalid sentence index')
        
        # if sentence index is the same as the index in the list
        try:
            if self._sentences[sentence_index].get_index() == sentence_index:
                return sentence_index

        except IndexError:
            pass       
        
        for sentence in self.get_sentences():
            if sentence.get_index() == sentence_index:
                list_index = self._sentences.index(sentence)
                break 

        return list_index

    def get_mapping_list_index_to_sentence_index(self, list_index):
        try:
            return self._sentences[list_index].get_index()
        except IndexError:
            return None

    def get_entity_start_end_indices(self):
        chunk_index = self.get_chunk_index()
        entity_start_indices = []
        entity_end_indices = []


        for sent in self._sentences:
            for entity in sent.get_entities():
                entity_sentence_index = entity.get_sentence_index()
                
                # compute start/end positions of entity in chunk
                in_chunk_sentence_offset = self.get_sentence_offset(entity_sentence_index)
                in_chunk_start_pos = in_chunk_sentence_offset + entity.get_start_pos()
                in_chunk_end_pos = in_chunk_sentence_offset + entity.get_end_pos()
                
                # add computed indices to list
                entity_start_indices.append([chunk_index, in_chunk_start_pos])
                entity_end_indices.append([chunk_index, in_chunk_end_pos])

        return entity_start_indices, entity_end_indices


    def get_entities(self):
        try:
            return self._entities
        except AttributeError:
            all_entities = []
            for sentence in self.get_sentences():
                all_entities.extend(sentence.get_entities())
            self._entities = all_entities
            
            return self._entities

            
    def __copy__(self):
        newone = type(self)(self._sentences, self._chunk_index)
        newone.__dict__.update(self.__dict__)
        return newone

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def add_sentence(self, sentence_index, new_sentence):
        sentence_indices = self.get_sentence_indices()

        if sentence_index < min(sentence_indices):
            self._sentences.insert(0, new_sentence)
        elif sentence_indices > max(sentence_indices):
            self._sentences.append(new_sentence)
        elif sentence_index in sentence_indices:
            print("DUPLICATE SENTENCE INDEX")
            self.set_sentence_by_index(sentence_index=sentence_index, new_sentence=new_sentence)
        
        self.update_sentence_offsets()