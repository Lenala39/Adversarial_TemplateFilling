import copy
import numpy as np
import itertools
import random
import numpy as np
import sys
from scipy import optimize
import copy
from ctro import *
from tqdm import tqdm

def align_groups(gt_groups, predicted_groups, used_slots=None):
    num_groups_diff = abs(len(gt_groups) - len(predicted_groups))
    if num_groups_diff > 0:
        if len(gt_groups) < len(predicted_groups):
            smaller_groups_list = gt_groups
        else:
            smaller_groups_list = predicted_groups
            
        for i in range(num_groups_diff):
            smaller_groups_list.append(Group())
            
    num_groups = len(gt_groups)
    
    
    # compute cost matrix
    cost_matrix = np.zeros((num_groups,num_groups))
    
    for gt_index in range(num_groups):
        for predicted_index in range(num_groups):
            stats_collection = F1StatisticsCollection()
            stats_collection.update(gt_groups[gt_index], predicted_groups[predicted_index], used_slots)
            cost_matrix[gt_index, predicted_index] = -stats_collection.get_micro_stats().f1()
            
    # solve linear asiignmnet problem
    optimal_gt_indices, optimal_predicted_indices = optimize.linear_sum_assignment(cost_matrix)
    group_pairs = []
    
    for i in range(num_groups):
        group_pairs.append((gt_groups[optimal_gt_indices[i]], predicted_groups[optimal_predicted_indices[i]]))
        
    return group_pairs



class AtomicEntitiesConfiguration:
    def __init__(self, num_groups, entity_compatibility_collection):
        self.free_entities = set()
        for entity in entity_compatibility_collection.get_entities():
            self.free_entities.add(entity.get_global_entity_index())

        self.num_groups = num_groups
        
        self.entity_compatibility_collection = entity_compatibility_collection
        
        # dict: old index -> zero based group index
        self.entity_assignments = {}
        
        
    
    # group index is zero based    
    def get_entity_indices_of_group(self, group_index):
        return [k for k,v in self.entity_assignments.items() if v == group_index]
    
    
    def get_num_free_entities(self):
        return len(self.free_entities)
    
    
    
    def assign_entity_to_group(self, entity_index, group_index):
        self.free_entities.remove(entity_index)
        self.entity_assignments[entity_index] = group_index
        
        
        
    def move_entity(self, entity_index, new_group_index):
        self.entity_assignments[entity_index] = new_group_index
    
    
    
    def create_seed_assignment(self):
        for group_index in range(self.num_groups):
            if len(self.free_entities) == 0:
                return
            
            entity_index = random.choice(list(self.free_entities))
            self.assign_entity_to_group(entity_index, group_index)
            
            
            
    def get_successor_configurations(self):
        successor_configurations = []
        
        for free_entity_index in self.free_entities:
            for group_index in range(self.num_groups):
                successor_configuration = copy.deepcopy(self)
                successor_configuration.assign_entity_to_group(free_entity_index, group_index)
                successor_configurations.append(successor_configuration)
                
        return successor_configurations
    
    
    
    def get_mixing_successor_configurations(self):
        successor_configurations = []
        
        for source_group_index in range(self.num_groups):
            source_group_entity_indices = self.get_entity_indices_of_group(source_group_index)
            if len(source_group_entity_indices) < 2:
                continue
                
            for dest_group_index in range(self.num_groups):
                if source_group_index == dest_group_index:
                    continue
                
                for entity_index in source_group_entity_indices:
                    successor_configuration = copy.deepcopy(self)
                    successor_configuration.move_entity(entity_index, dest_group_index)
                    successor_configurations.append(successor_configuration)
                    
        return successor_configurations
        
        
        
    def assign_entities_randomly(self):
        for group_index in range(self.num_groups):
            if len(self.free_entities) == 0:
                break
            
            entity_index = random.choice(list(self.free_entities))
            self.assign_entity_to_group(entity_index, group_index)
            
        for entity_index in list(self.free_entities):
            group_index = random.randint(0, self.num_groups-1)
            self.assign_entity_to_group(entity_index, group_index)
    
    
    
    def get_entity_pair_compatibility(self, entity1_index, entity2_index):
        return self.entity_compatibility_collection.get_entity_pair_compatibility(entity1_index, entity2_index)
    
    
    
    def get_entity(self, entity_index):
        for entity in self.entity_compatibility_collection.get_entities():
            if entity.get_global_entity_index() == entity_index:
                return entity
            
        raise IndexError('ERROR: entity not found')
    
    
    
    def get_inter_entity_pairs(self):
        entity_pairs = set()
        
        for inner_group_index in range(self.num_groups):
            inner_group_entities = self.get_entity_indices_of_group(inner_group_index)
            
            for inner_group_entity in inner_group_entities:
                for outer_group_index in range(self.num_groups):
                    if inner_group_index == outer_group_index:
                        continue
                    
                    outer_group_entities = self.get_entity_indices_of_group(outer_group_index)
                    
                    for outer_group_entity in outer_group_entities:
                        entity_pair = sorted((inner_group_entity, outer_group_entity))
                        entity_pairs.add(tuple(entity_pair))
                        
        return entity_pairs
    
    
    
    def compute_positive_score(self):
        values = []
        
        for group_index in range(self.num_groups):
            group_entity_indices = self.get_entity_indices_of_group(group_index)
            
            if len(group_entity_indices) == 1:
                values.append(0.5)
                continue
                
            index_pairs = itertools.combinations(group_entity_indices, 2)
            
            for entity_index1,entity_index2 in index_pairs:
                score = self.get_entity_pair_compatibility(entity_index1, entity_index2) 
                values.append(score)
                
        # return sum of scores normalized by number of pairs
        array = np.array(values)
        return np.sum(array) / len(array)
    
    
    
    def compute_negative_score(self):
        values = []
        
        for inner_entity,outer_entity in self.get_inter_entity_pairs():
            values.append(-self.get_entity_pair_compatibility(inner_entity, outer_entity))
        
        
        # return sum of scores normalized by number of pairs
        array = np.array(values)
        return np.sum(array) / len(array)
    
    
    
    def compute_total_score(self, include_inter_compatibility):
        if include_inter_compatibility:
            return self.compute_positive_score() + self.compute_negative_score()
        else:
            return self.compute_positive_score()
            
        
        
    def create_groups(self):
        groups = []
        
        for i in range(self.num_groups):
            entity_indices = self.get_entity_indices_of_group(i)
            group = Group()
            
            for entity_index in entity_indices:
                entity = self.get_entity(entity_index)
                slot_name = list(entity.get_referencing_slot_names())[0]
                
                slot_value = SlotValue()
                slot_value.string = ' '.join(entity.get_tokens())
                
                if slot_name not in group.slots:
                    group.slots[slot_name] = []
                    
                group.slots[slot_name].append(slot_value)
                
            groups.append(group)
            
        return groups
    
    
    
    def print_configuration(self):
        for group_index in range(self.num_groups):
            entity_indices = self.get_entity_indices_of_group(group_index)
            print(group_index, entity_indices)
            
      
        
        
            
def sort_configurations(configurations, include_inter_compatibility=False):
    return sorted(configurations, key=lambda conf:conf.compute_total_score(include_inter_compatibility), reverse=True)



        
class ConfigurationOptimizer:
    def __init__(self, num_groups, entity_compatibility_collection):
        self.initial_configuration = AtomicEntitiesConfiguration(num_groups, entity_compatibility_collection)
        self.slot_constraints_validator = None
        
        
    
    def set_slot_constraints_validator(self, slot_constraints_validator):
        self.slot_constraints_validator = slot_constraints_validator
        
        
        
    def filter_invalid_configurations(self, configurations):
        if self.slot_constraints_validator is None:
            return configurations
        
        result_configurations = []
        
        for configuration in configurations:
            # create groups of configuration
            groups = configuration.create_groups(self.annotations, self.annotation_tokens, self.slot_prediction_dict)
            
            valid = True
            for group in groups:
                if not self.slot_constraints_validator.validate_single_slot_assignments(group):
                    valid = False
                    break
                
            if valid:
                result_configurations.append(configuration)
                
        return result_configurations
        
        
        
    def create_random_configuration(self):
        configuration = copy.deepcopy(self.initial_configuration)
        configuration.assign_entities_randomly()
        return configuration
        
        
    def beam_search(self, beam_size, include_inter_compatibility=False):
        initial_configuration = copy.deepcopy(self.initial_configuration)
        
        if initial_configuration.num_groups == 1:
            for entity_index in list(initial_configuration.free_entities):
                initial_configuration.assign_entity_to_group(entity_index, 0)
                
            return initial_configuration
        
        # create seed configurations
        beam = []
        
        for i in range(beam_size):
            seed_configuartion = copy.deepcopy(initial_configuration)
            seed_configuartion.create_seed_assignment()
            beam.append(seed_configuartion)
            
        num_free_entities = beam[0].get_num_free_entities()
        
        while num_free_entities > 0:
            # get successor configurations for all configurations in current beam
            successor_configurations = []
                
            for configuration in beam:
                successor_configurations.extend(configuration.get_successor_configurations())
                
            # filter successor configurations
            successor_configurations = self.filter_invalid_configurations(successor_configurations)
            if len(successor_configurations) == 0:
                break
            
            successor_configurations = sort_configurations(successor_configurations, include_inter_compatibility)
                
            if len(successor_configurations) <= beam_size:
                beam = successor_configurations
                num_free_entities -= 1
                continue
                
            beam = successor_configurations[:beam_size]
            num_free_entities -= 1

        return beam[0]


'''
num_groups = 3
num_entities = 6

new_indices = { index:index for index in range(num_entities)}
compatibility_matrix = np.zeros((num_entities, num_entities))

compatibility_matrix[0,1] = 1
compatibility_matrix[3,4] = 1
compatibility_matrix[3,5] = 1
compatibility_matrix[4,5] = 1



#optimizer = ConfigurationOptimizer(num_groups, new_indices, compatibility_matrix)
#best_configuration = optimizer.beam_search(10)
#best_configuration.print_configuration()




test_configuration = AtomicEntitiesConfiguration(num_groups, new_indices, compatibility_matrix)
#test_configuration.create_initial_assignment()
test_configuration.assign_entities_randomly()
test_configuration.print_configuration()
'''
