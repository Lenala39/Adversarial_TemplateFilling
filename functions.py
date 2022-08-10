from ctro import *


def select_entities_by_template_type(entities, template_type):
    template_slots = ontology.group_slots[template_type]
    
    return [entity for entity in entities if len(template_slots & entity.get_referencing_slot_names()) > 0]



def print_group(group, slots):
    slots = sorted(slots)
    
    for slot_name in slots:
        if slot_name in group.slots:
            slot_fillers = []
            
            for slot_value in group.slots[slot_name]:
                slot_fillers.append(slot_value.string.replace(' ##', ''))
                
            print(slot_name + ': ' + ' | '.join(slot_fillers))
        


def print_groups_dict(groups_dict, used_slots):
    for group_name in ontology.used_group_names:
        if group_name in groups_dict:
            groups = groups_dict[group_name]
            print('Template Type: ', group_name, ' ===========================')
            for i,group in enumerate(groups):
                print('Instance ', i+1)
                print_group(group, used_slots)
                print()
                print('-----------')