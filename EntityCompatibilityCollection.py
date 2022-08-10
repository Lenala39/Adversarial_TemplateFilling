from multiprocessing.sharedctypes import Value
import sys
import itertools
from Entity import *
from sklearn.metrics import f1_score

class EntityCompatibilityCollection:
    def __init__(self, entities):
        self._entities = entities
        self._entity_pair_compatibilities = None
        self._entity_pair_compatibilities_NO_SIGMOID = None
        
        # all unordered entity pairs; sorted by global entity index
        entity_pairs = itertools.combinations(entities, 2)
        entity_pairs = sorted(entity_pairs, key=lambda pair: (pair[0].get_global_entity_index(), pair[1].get_global_entity_index()))
        self._entity_pairs = entity_pairs
        
        
    def get_entities(self):
        return self._entities
    
    
    def get_entity_by_index(self, entity_index):
        for entity in self.get_entities():
            if entity.get_global_entity_index() == entity_index:
                return entity
            
        raise IndexError('Entity not found')
        
        
    def get_entity_pairs(self):
        return self._entity_pairs
        
    def get_entity_pair_compatibilities(self):
        return self._entity_pair_compatibilities

    def get_entity_pair_compatibilities_no_sigmoid(self):
        return self._entity_pair_compatibilities_NO_SIGMOID

        
    def get_ground_truth_compatibility_list(self):
        gt_compatibilities = []
        
        for entity_pair in self.get_entity_pairs():
            if len(entity_pair[0].get_referencing_template_ids() & entity_pair[1].get_referencing_template_ids()) > 0: #set comparison
                compatibility = 1
            else:
                compatibility = 0
                
            gt_compatibilities.append(compatibility)
        
        # # print nicely for get_ground_truth_compatibility_list(self) in EntityCompatibilityCollection
        # temp = list(zip(gt_compatibilities, self.get_entity_pairs()))
        # for elem in temp:
            # print(f"Compatibility: {elem[0]} between {elem[1][0].get_tokens()} and {elem[1][1].get_tokens()} with labels {elem[1][0].get_label()} and {elem[1][1].get_label()}")
        return gt_compatibilities
    
    
    
    def get_entity_indices_of_pairs(self):
        entity1_indices = []
        entity2_indices = []
        
        for entity1,entity2 in  self.get_entity_pairs():
            entity1_indices.append(entity1.get_global_entity_index())
            entity2_indices.append(entity2.get_global_entity_index())
            
        return entity1_indices, entity2_indices
    
    def compute_compatibility_score(self):
        try:
            if not self._entity_pair_compatibilities:
                # print("ERROR: entity pair compatibilities not set in EnityCompatibilityCollection.compute_compatibility_score")
                return None 
        except ValueError: # self._entity_pair_comp is not None
            pass 
        
        mapped_entity_comp = [0 if elem < 0.5 else 1 for elem in self._entity_pair_compatibilities]
        gt_comp = self.get_ground_truth_compatibility_list()
        f1 = round(f1_score(y_true=gt_comp, y_pred=mapped_entity_comp, average="micro"),2)

        return f1

    def set_entity_pair_compatibilities(self, compatibilities_list):
        assert len(self.get_entity_pairs()) == len(compatibilities_list)
        self._entity_pair_compatibilities = compatibilities_list
    
    def set_entity_pair_compatibilities_no_sigmoid(self, compatibilities_list):
        assert len(self.get_entity_pairs()) == len(compatibilities_list)
        self._entity_pair_compatibilities_NO_SIGMOID = compatibilities_list
        
        
    def get_entity_pair_compatibility(self, entity1_index, entity2_index):
        '''
        entity1 = self.get_entity_by_index(entity1_index)
        entity2 = self.get_entity_by_index(entity2_index)
        if len(entity1.get_referencing_template_ids() & entity2.get_referencing_template_ids()) > 0:
            return 1
        else:
            return 0
        '''
        
        if self._entity_pair_compatibilities is None:
            raise('ERROR: entity pair compatibilities not set')
            
        query_indices_pair = sorted((entity1_index, entity2_index))
        pairs_set = set()
        
        for i,entity_pair in enumerate(self.get_entity_pairs()):
            indices_pair = [entity_pair[0].get_global_entity_index(), entity_pair[1].get_global_entity_index()]
            pairs_set.add(tuple(indices_pair))
            
            if indices_pair == query_indices_pair:
                return self._entity_pair_compatibilities[i]
        print(pairs_set)
        print(query_indices_pair)
        print('--')
        return 0
        raise Exception('Entity pair not found')
        
        
        
    def print_out(self):
        print('indices of entity_pairs')
        
        for entity_pair in self._entity_pairs:
            print((entity_pair[0].get_global_entity_index(), entity_pair[1].get_global_entity_index()))