from SlotFillingCompModule import *

from SlotFillingCompModule import * 
from stats_printer import * 
from helper_functions import * 

disease_str, victim_type, start, end, comp_mode, data_mode, model = parse_cli_arguments(for_slot_filling=True)

if model.lower() == "clare":
    model = model.upper() 
else:
    model=model.capitalize() 

if disease_str == "gl":
    model_prefix = "glaucoma"
else:
    model_prefix = disease_str

print(f"{model} {disease_str} {victim_type} {comp_mode} {data_mode}")


load_slot_indices = False 
model_name = None 

slot_filling_module = prepare_module(model_prefix=model_prefix, load_from_pickle=True, train=False, load_weights=False, model=model, 
                                        comp_mode=comp_mode, load_augmented=False, victim_type=victim_type, 
                                        load_slot_indices=load_slot_indices, model_name=model_name)

import_attacked(disease_str=disease_str, model=model, victim_type=victim_type, comp_mode=comp_mode, 
                        slot_filling_module=slot_filling_module, as_attacked=True) 

if victim_type == "comp":
    attack_log_path = os.path.join(dir_path, "attack_logs/", f"{model.capitalize()}/", f"{victim_type.capitalize()}/", f"{model}_{disease_str}_{victim_type}_{data_mode}_{comp_mode}_{start}to{end}_results.csv")
else:
    attack_log_path = os.path.join(dir_path, "attack_logs/", f"{model.capitalize()}/", f"{victim_type.capitalize()}/", f"{model}_{disease_str}_{victim_type}_{data_mode}_{start}to{end}_results.csv")

slot_filling_module.augment_dataset(attack_log_path=attack_log_path, test_or_train=data_mode, append=False)

if data_mode.lower() == "train":
    save_modified_data(attacked_doc_chunkings=slot_filling_module.document_chunkings_train, 
                    attacked_batches=slot_filling_module.batches_train, 
                    attacked_documents=slot_filling_module.documents_train,
                    disease_str=disease_str, comp_mode=comp_mode, model=model, victim_type=victim_type, 
                    data_mode=data_mode, further_descr="augmented_only2", 
                    subfolder="Augmented")
else:
    save_modified_data(attacked_doc_chunkings=slot_filling_module.document_chunkings_test, 
                    attacked_batches=slot_filling_module.batches_test, 
                    attacked_documents=slot_filling_module.documents_test,
                    disease_str=disease_str, comp_mode=comp_mode, model=model, victim_type=victim_type, 
                    data_mode=data_mode, further_descr="augmented_only2", 
                    subfolder="Augmented")
