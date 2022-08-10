import pickle 
import os
from Document import * 
from ctro import Abstract 

def convert_abstracts_to_documents(abstracts, tokenizer):
    documents = []

    for abstract in abstracts:
        doc = Document()
        doc.set_from_abstract(abstract, tokenizer)
        documents.append(doc)

        for annotation in abstract.annotated_abstract.annotations:
            annotation.tokens = tokenizer.tokenize(' '.join(annotation.tokens))

        groups = abstract.group_collection.groups
        for group_name in groups:
            for group in groups[group_name]:
                for slot_name in group.slots:
                    for slot_value in group.slots[slot_name]:
                        slot_value_string = ' '.join(
                            slot_value.annotation.tokens)
                        slot_value.string = slot_value_string

    return documents


def import_abstracts(disease_string, path, train_ids, test_ids, val_ids):
    train_abstracts = []
    test_abstracts = []
    val_abstracts = []
    
    if train_ids:
        for abstract_id in train_ids:
            abstract = Abstract()
            abstract.abstract_id = abstract_id
            abstract.import_data(disease_string, abstract_id, path)
            train_abstracts.append(abstract)
    
    if test_ids:    
        for abstract_id in test_ids:
                abstract = Abstract()
                abstract.abstract_id = abstract_id
                abstract.import_data(disease_string, abstract_id, path)
                test_abstracts.append(abstract)
    if val_ids:    
        for abstract_id in val_ids:
                abstract = Abstract()
                abstract.abstract_id = abstract_id
                abstract.import_data(disease_string, abstract_id, path)
                val_abstracts.append(abstract)
            
    return train_abstracts, test_abstracts, val_abstracts

class DataImporter:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer 

        
    def import_documents_dump(self, dump_file_path):
        f = open(dump_file_path, 'rb')
        abstracts_train, abstracts_validation, abstracts_test = pickle.load(f, encoding='latin1')
        f.close()

        documents_train = convert_abstracts_to_documents(abstracts_train, self.tokenizer)
        documents_validation = convert_abstracts_to_documents(abstracts_validation, self.tokenizer)
        documents_test = convert_abstracts_to_documents(abstracts_test, self.tokenizer)

        return documents_train, documents_test, documents_validation

    def import_from_pickle(self, disease_str, dump_file_path):
        print("Importing Documents")
        documents = self.__import_documents_from_pickle(disease_str, dump_file_path)
        print("Importing Doc Chunkings")
        document_chunkings = self.__import_doc_chunkings_from_pickle(disease_str, dump_file_path)
        print("Importing Batches")
        batches = self.__import_batches_from_pickle(disease_str, dump_file_path)
        
        return documents, document_chunkings, batches 

    def import_augmented(self, disease_str, model, victim_type, comp_mode, dump_file_path):
        # DOCUMENTS
        if victim_type == "comp":
            #filename_docs = os.path.join(dump_file_path, f"{model}_{disease_str}_{victim_type.lower()}_train_{comp_mode}_documents_augmented.pickle")
            filename_docs = os.path.join(dump_file_path, f"{model}_{disease_str}_{victim_type.lower()}_train_{comp_mode}_all_augmented_documents_attacked.pickle")
        else:
            #filename_docs = os.path.join(dump_file_path, f"{model}_{disease_str}_{victim_type.lower()}_train_documents_augmented.pickle")
            filename_docs = os.path.join(dump_file_path, f"{model}_{disease_str}_{victim_type.lower()}_train_all_augmented_documents_attacked.pickle")

        print("Importing augmented Documents")
        try:
            with open(filename_docs, "rb") as f:
                documents_train = pickle.load(f)
        except FileNotFoundError:
            documents_train = None 
        try:
            with open(filename_docs.replace("train", "test"), "rb") as f:
                documents_test = pickle.load(f)
        except FileNotFoundError:
            documents_test = None 

        documents = [documents_train, documents_test]

        # DOC CHUNKINGS
        if victim_type == "comp":
            filename_dc = os.path.join(dump_file_path, f"{model}_{disease_str}_{victim_type.lower()}_train_{comp_mode}_all_augmented_doc_chunkings_attacked.pickle")
        else:
            filename_dc = os.path.join(dump_file_path, f"{model}_{disease_str}_{victim_type.lower()}_train_all_augmented_doc_chunkings_attacked.pickle")

        print("Importing augmented Doc Chunkings")

        try:
            with open(filename_dc, "rb") as f:
                doc_chunkings_train = pickle.load(f)
        except FileNotFoundError:
            doc_chunkings_train = None
            
        try:            
            with open(filename_dc.replace("train", "test"), "rb") as f:
                doc_chunkings_test = pickle.load(f)
        except FileNotFoundError:
            doc_chunkings_test = None 

        doc_chunkings = [doc_chunkings_train, doc_chunkings_test]
        

        # BATCHES
        if victim_type == "comp":
            # filename_b = os.path.join(dump_file_path, f"{model}_{disease_str}_{victim_type.lower()}_train_{comp_mode}_batches_augmented.pickle")
            filename_b = os.path.join(dump_file_path, f"{model}_{disease_str}_{victim_type.lower()}_train_{comp_mode}_all_augmented_batches_attacked.pickle")
        else:
            # filename_b = os.path.join(dump_file_path, f"{model}_{disease_str}_{victim_type.lower()}_train_batches_augmented.pickle")
            filename_b = os.path.join(dump_file_path, f"{model}_{disease_str}_{victim_type.lower()}_train_all_augmented_batches_attacked.pickle")
        
        print("Importing augmented Batches")
        try:
            with open(filename_b, "rb") as f:
                batches_train = pickle.load(f)
        except FileNotFoundError:
            batches_train = None
        
        try:
            with open(filename_b.replace("train", "test"), "rb") as f:
                batches_test = pickle.load(f)
        except FileNotFoundError:
            batches_test = None 
            
        batches = [batches_train, batches_test]

        return documents, doc_chunkings, batches

    def __import_doc_chunkings_from_pickle(self, disease_str, dump_file_path):
        
        filename = f"{disease_str}_doc_chunkings_train.pickle"
        with open(os.path.join(dump_file_path, filename), "rb") as f:
            document_chunkings_train = pickle.load(f)
        
        filename = f"{disease_str}_doc_chunkings_test.pickle"
        with open(os.path.join(dump_file_path, filename), "rb") as f:
            document_chunkings_test = pickle.load(f)
        
        filename = f"{disease_str}_doc_chunkings_validation.pickle"
        with open(os.path.join(dump_file_path, filename), "rb") as f:
            document_chunkings_validation = pickle.load(f)

        return document_chunkings_train, document_chunkings_test, document_chunkings_validation

    def __import_batches_from_pickle(self, disease_str, dump_file_path):

        filename = f"{disease_str}_batches_train.pickle"
        with open(os.path.join(dump_file_path, filename), "rb") as f:
            batches_train = pickle.load(f)
        
        filename = f"{disease_str}_batches_test.pickle"
        with open(os.path.join(dump_file_path, filename), "rb") as f:
            batches_test = pickle.load(f)
        
        filename = f"{disease_str}_batches_validation.pickle"
        with open(os.path.join(dump_file_path, filename), "rb") as f:
            batches_validation = pickle.load(f)    
        
        return batches_train, batches_test, batches_validation
    
    def __import_documents_from_pickle(self, disease_str, dump_file_path):

        filename = f"{disease_str}_documents_train.pickle"
        with open(os.path.join(dump_file_path, filename), "rb") as f:
            documents_train = pickle.load(f)
        
        filename = f"{disease_str}_documents_test.pickle"
        with open(os.path.join(dump_file_path, filename), "rb") as f:
            documents_test = pickle.load(f)
        
        filename = f"{disease_str}_documents_validation.pickle"
        with open(os.path.join(dump_file_path, filename), "rb") as f:
            documents_validation = pickle.load(f)

        return documents_train, documents_test, documents_validation

    def import_attacked(self, model, disease_str, victim_type, comp_mode, dump_file_path):
        dump_file_path_model_task = os.path.join(dump_file_path, f"{model.capitalize()}/", f"{victim_type.capitalize()}/")
        
        document_chunkings_train_attacked = None 
        batches_train_attacked = None 

        document_chunkings_test_attacked = None 
        batches_test_attacked = None 

        if not comp_mode:
            relevant_files = [f for f in os.listdir(dump_file_path_model_task) if (disease_str in f and model in f and victim_type in f)] 
        else:
            relevant_files = [f for f in os.listdir(dump_file_path_model_task) if (disease_str in f and model in f and victim_type in f and comp_mode in f)] 
        
        for relevant_f in relevant_files:
            with open(os.path.join(dump_file_path_model_task, relevant_f), "rb") as in_file:
                data = pickle.load(in_file)

            if "train" in relevant_f:
                if "doc_chunking" in relevant_f:
                    document_chunkings_train_attacked = data 
                    
                elif "batches" in relevant_f:
                    batches_train_attacked = data 
                    
            elif "test" in relevant_f:
                if "doc_chunking" in relevant_f:
                    document_chunkings_test_attacked = data
                
                elif "batches" in relevant_f:
                    batches_test_attacked = data
        
        if isinstance(batches_test_attacked[0], list):
            # flatten
            batches_test_attacked = [batch for doc in batches_test_attacked for batch in doc]
            batches_train_attacked = [batch for doc in batches_train_attacked for batch in doc]

        return document_chunkings_train_attacked, document_chunkings_test_attacked, batches_train_attacked, batches_test_attacked
        
    def import_documents(self, disease_str, only_train=True):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        ANNOTATED_DATA_DIR = os.path.join(dir_path, "Data/", "AnnotatedCorpus/")
        train_ids, test_ids, val_ids = self.return_train_test_val_ids(
            disease_str=disease_str)
        if only_train:
            test_ids = None
            val_ids = None

        abstracts_train, abstracts_test, abstracts_validation = import_abstracts(disease_string=disease_str, path=ANNOTATED_DATA_DIR,
                                                                                 train_ids=train_ids, test_ids=test_ids, val_ids=val_ids)

        documents_train = convert_abstracts_to_documents(abstracts_train, self.tokenizer)
        documents_validation = convert_abstracts_to_documents(abstracts_validation, self.tokenizer)
        documents_test = convert_abstracts_to_documents(abstracts_test, self.tokenizer)

        return documents_train, documents_test, documents_validation
        
    def return_train_test_val_ids(self, disease_str):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        split_folder = os.path.join(dir_path, "Data/", "dataset_splits")

        if disease_str == "gl":
            disease_str = "glaucoma"

        with open(os.path.join(split_folder, f"{disease_str}_train_ids.txt"), "r") as train_file:
            train_ids = train_file.readlines()
            train_ids = [elem.replace("\n", "") for elem in train_ids]

        with open(os.path.join(split_folder, f"{disease_str}_test_ids.txt"), "r") as test_file:
            test_ids = test_file.readlines()
            test_ids = [elem.replace("\n", "") for elem in test_ids]

        with open(os.path.join(split_folder, f"{disease_str}_validation_ids.txt"), "r") as val_file:
            val_ids = val_file.readlines()
            val_ids = [elem.replace("\n", "") for elem in val_ids]

        return train_ids, test_ids, val_ids
