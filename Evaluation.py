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
if victim_type.lower() == "extraction":
    attack_log_path = f"/home/lena/TemplateFilling/attack_logs/{model.capitalize()}/{victim_type.capitalize()}/{model}_{disease_str}_{victim_type.lower()}_{data_mode.lower()}_{start}to{end}_results.csv"
else:
    attack_log_path = f"/home/lena/TemplateFilling/attack_logs/{model.capitalize()}/{victim_type.capitalize()}/{model}_{disease_str}_{victim_type.lower()}_{data_mode.lower()}_{comp_mode.lower()}_{start}to{end}_results.csv"

robust_model = True
load_augmented = False
load_attacked = False

if model.lower() == "araujo" and robust_model:
    load_slot_indices = False
else:
    load_slot_indices = True 

epochs = 30

load_normal = True if (not load_attacked and not load_augmented) else False 
print(f"ROBUST MODEL: {robust_model} \nAUGMENTED DATA: {load_augmented} \nATTACKED DATA: {load_attacked} \nNORMAL: {load_normal}")

if load_augmented and load_attacked:
    print("STOP, need to select augmented **or** attacked")
    exit() 

if load_augmented:
    data_modifier = "augmented_only2_"
elif load_attacked:
    data_modifier = "attacked_"
elif not load_attacked and not load_augmented:
    data_modifier = ""


model_name = None 

model_modifier = ""
if robust_model:
    model_modifier = "robust_"

    if victim_type == "comp":
        model_name = f"{model_prefix}_{model}_{victim_type}_{comp_mode}_augmented_30epochs"
    else:
        model_name = f"{model_prefix}_{model}_{victim_type}_augmented_30epochs"
    

else:
    model_name = f"{model_prefix}_{epochs}epochs"

slot_filling_module = prepare_module(model_prefix=model_prefix, load_from_pickle=True, train=False, load_weights=True, model=model, 
                                        comp_mode=comp_mode, load_augmented=True, victim_type=victim_type, 
                                        load_slot_indices=load_slot_indices, model_name=model_name)

if load_attacked:
    import_attacked(disease_str=disease_str, model=model, victim_type=victim_type, comp_mode=comp_mode, 
                        as_attacked=False, slot_filling_module=slot_filling_module)

elif load_augmented:
    filepath_pickles = os.path.join(dir_path, "Data/", "Augmented/", "Only_Augmented_With_Unchanged/")
    documents, document_chunkings, batched = slot_filling_module.data_importer.import_augmented(disease_str, model, victim_type, comp_mode, filepath_pickles)

    slot_filling_module.documents_train = documents[0]
    slot_filling_module.documents_test = documents[1]

    slot_filling_module.document_chunkings_train = document_chunkings[0]
    slot_filling_module.document_chunkings_test = document_chunkings[1]

    document_encoder = DocumentEncoder(tokenizer, slot_filling_module.slot_indices, 'no_slot')
    if slot_filling_module.documents_train:
        slot_filling_module.batches_train = create_batches(slot_filling_module.documents_train, slot_filling_module.document_chunkings_train, document_encoder, slot_filling_module.slot_indices)
    if slot_filling_module.document_chunkings_test:
        slot_filling_module.batches_test = create_batches(slot_filling_module.documents_test, slot_filling_module.document_chunkings_test, document_encoder, slot_filling_module.slot_indices)
            

print(f"Number of test documents: {len(slot_filling_module.documents_test)}")

if victim_type == "extraction":
    print("EVENT EXTRACTION")
    print("Computing eval for event extraction")
    filename = f"{disease_str}_{model}_{data_modifier}{model_modifier}EventExtraction_results.tex"
    slot_filling_module.evaluate_entity_prediction(filename=filename)


else: 
    print("COMPATIBILITY SCORE")
    print("Call encoder on the test set")
    slot_filling_module.call_encoder_test_set()
    beam_size = 10
    print(f"Computing eval for slot filling with beam size {beam_size}")
    filename = f"{disease_str}_{model}_{data_modifier}{model_modifier}SlotFilling_comp_{comp_mode}_results.tex"
    slot_filling_module.evaluate_slot_filling(beam_size=beam_size, filename=filename)

exit()
if victim_type == "extraction":
    print("EVENT EXTRACTION")
    print("Computing eval for event extraction")
    print(f"{model}_{disease_str}_{victim_type}_{comp_mode}_{data_modifier}{model_modifier}{epochs}epochs_results.tex")
    slot_filling_module.evaluate_entity_prediction(filename=f"{model}_{disease_str}_{victim_type}_{comp_mode}_{data_modifier}{model_modifier}{epochs}epochs_results.tex")

else:
    print("COMPATIBILITY SCORE")
    print("Call encoder on the test set")
    slot_filling_module.call_encoder_test_set()
    beam_size = 10
    print(f"Computing eval for slot filling with beam size {beam_size}")
    slot_filling_module.evaluate_slot_filling(beam_size=beam_size, filename=f"{model}_{disease_str}_{victim_type}_{comp_mode}_{data_modifier}{model_modifier}{beam_size}BEAM_{epochs}epochs_results.tex")

#beam_size = 50
#print(f"Computing eval for slot filling with beam size {beam_size}")
#slot_filling_module.evaluate_slot_filling(beam_size=beam_size, filename=f"{model}_{disease_str}_{victim_type}_{comp_mode}_{modifier}{beam_size}BEAM_results.tex")


