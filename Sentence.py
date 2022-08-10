from Entity import *
from copy import deepcopy

class Sentence:
    def __init__(self, tokens=[], index=None):
        self._tokens = tokens
        self._index = index
        self._entities = []
        
        
    def clear(self):
        self._tokens = []
        self._index = None
        self._entities = []
        
        
    def __eq__(self, other):
        if not isinstance(other, Sentence):
            raise TypeError()
            
        if self.get_tokens() != other.get_tokens():
            return False
        if self.get_index() != other.get_index():
            return False
        if self.get_entities() != other.get_entities():
            return False
        
        return True
    
    
    def __neq__(self, other):
        return not self == other
        
        
    def get_tokens(self):
        return self._tokens
    
    
    def get_index(self):
        return self._index
    
    
    def set_index(self, index):
        self._index = index
    
    
    def get_entities(self):
        return self._entities
    
    
    def add_entity(self, entity):
        self._entities.append(entity)
    
    
    def get_num_tokens(self):
        return len(self._tokens)
    
    
    def get_num_entities(self):
        return len(self._entities)
    
    
    def print_out(self, print_entities=False):
        # print sentence index and tokens
        print('Sentence index:', self.get_index())
        sentence_string = ' '.join(self.get_tokens()).replace(" ##", "")
        print('Sentence string:', sentence_string)
        if print_entities:
            # print entities
            for entity in self._entities:
                entity.print_out()
                
            
    def set_from_annotated_sentence(self, annotated_sentence):
        sentence_index = annotated_sentence.sentence_id - 1
        self._index = sentence_index
        self._tokens = annotated_sentence.tokens.copy()
        self._entities = []
        
        # extract entities from AnnotatedSentence object
        for entity_index,annotation in enumerate(annotated_sentence.annotations):
            entity = Entity()
            start_index, end_index = annotated_sentence.entity_boundaries[entity_index]
                
            # set sentence index of entity
            entity.set_sentence_index(sentence_index)
                
            # set globbal index of entity
            entity.set_global_entity_index(annotation.global_index)
                
            # set in sentence boundaries of entity
            entity.set_start_pos(start_index)
            entity.set_end_pos(end_index)
                
            # extract tokens
            tokens = []
                
            for token_index in range(start_index, end_index+1):
                tokens.append(annotated_sentence.tokens[token_index])
                    
            entity.set_tokens(tokens)
                
            # most general superclass of entity
            entity.set_label(annotation.most_general_superclass)
                
            # estimate referencing slot names
            for (group_id,slot_name) in annotation.referencing_slots:
                entity.add_referencing_slot_name(slot_name)
                entity.add_referencing_template_ids(group_id)
                    
            # add entity to collection
            self.add_entity(entity)

    def set_tokens(self, tokens):
        self._tokens = []
        
        for token in tokens:
            self._tokens.append(token)
    
    def get_string(self):
        return " ".join(self._tokens).replace(" ##", "")

    def __copy__(self):
        newone = type(self)()
        newone.__dict__.update(self.__dict__)
        return newone


    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def set_entities(self, new_entities):
        self._entities = []
        for ent in new_entities:
            self._entities.append(ent)

    