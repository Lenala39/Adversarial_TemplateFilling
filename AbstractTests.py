import unittest
import sys
from Document import *
from SlotFillingCompModule import *



class AbstractTests(unittest.TestCase):
    def setUp(self):
        slot_filling_module = SlotFillingCompModule()
        slot_filling_module.import_documents(dump_file_path)
        slot_filling_module.create_slot_indices(1)
        
        self.slot_filling_module = slot_filling_module
        self.documents = slot_filling_module.documents_train #+ slot_filling_module.documents_validation + slot_filling_module.documents_test

        '''
        document = self.documents[11]
        for entity in document.get_entity_collection():
            print(str(entity.tokens) + ' : ' + entity.get_label() + ' : ' + str(entity.get_referencing_slot_names()) )
        '''


    def test_sentence_extraction(self):
        for document in self.documents:
            abstract = document.get_abstract()
            
            sentence_ids = sorted(abstract.get_sentence_numbers())
            
            for sentence_id in sentence_ids:
                sentence1 = tokenizer.tokenize(' '.join(abstract.get_sentence_tokens(sentence_id)))
                sentence2 = document.get_sentence_tokens(sentence_id - 1)
                
                self.assertEqual(sentence1, sentence2)
                
                
                
    def test_chunking(self):
        for document in self.documents:
            chunking = document.create_chunking(256)
            
            chunking_sentences = []
            for chunk in chunking:
                chunking_sentences.extend(chunk.get_sentences())
                
            self.assertEqual(document.get_num_sentences(), len(chunking_sentences))
            
            for i in range(len(chunking_sentences)):
                doc_sentence = document.get_sentence(i)
                chunk_sentence = chunking_sentences[i]
                self.assertEqual(doc_sentence, chunk_sentence)
                
                
                
    def test_entity_tokens(self):
        for document in self.documents:
            chunking = document.create_chunking(256)
            entity_collection = document.get_entity_collection()
            
            entity_tokens = entity_collection.get_all_entity_tokens()
            entity_tokens_document = entity_collection.get_all_entity_tokens_from_document(document)
            entity_tokens_chunking = entity_collection.get_all_entity_tokens_from_document_chunking(chunking)
            
            self.assertEqual(entity_tokens, entity_tokens_document)
            self.assertEqual(entity_tokens_document, entity_tokens_chunking)
            
            
            
    def test_entity_label_encoding(self):
        for document in self.documents:
            # information from document
            chunking = document.create_chunking(256)
            entity_collection = document.get_entity_collection().filter_by_referencing_slots_names({'hasAuthor'})
            
            # slot names of entities
            all_referenced_entity_tokens = []
            all_referencing_slot_names = []
            
            
            for entity in entity_collection:
                slot_list = list(entity.get_referencing_slot_names())
                
                if len(slot_list) > 0:
                    all_referenced_entity_tokens.append(entity.get_tokens())
                    all_referencing_slot_names.append(slot_list[0])
            
            '''
            assert len(all_referenced_entity_tokens) == len(all_referencing_slot_names)
            for i in range(len(all_referenced_entity_tokens)):
                print(all_referenced_entity_tokens[i], all_referencing_slot_names[i])
                
            sys.exit()
            '''
            
            # indices
            slot_indices = self.slot_filling_module.slot_indices
            slot_indices_reverse = self.slot_filling_module.slot_indices_reverse
            
            start_tokens_from_doc = [tokens[0] for tokens in all_referenced_entity_tokens]
            end_tokens_from_doc = [tokens[-1] for tokens in all_referenced_entity_tokens]
            tokens_from_doc = [tokens for tokens in all_referenced_entity_tokens]
            
            start_tokens_encoding, end_tokens_encoding = entity_collection.encode_entity_start_end_positions_by_slot_names(slot_indices, chunking)
            
            start_tokens_from_encoding = []
            end_tokens_from_encoding = []
            
            start_labels_from_encoding = []
            end_labels_from_encoding = []
            
            # get tokens
            for chunk in chunking:
                chunk_index = chunk.get_chunk_index()
                
                for in_chunk_token_offset in range(len(chunk)):
                    current_chunk_token = chunk.get_token(in_chunk_token_offset)
                    
                    start_encoding = start_tokens_encoding[chunk_index, in_chunk_token_offset]
                    end_encoding = end_tokens_encoding[chunk_index, in_chunk_token_offset]
                    
                    start_slot_name = slot_indices_reverse[start_encoding]
                    end_slot_name = slot_indices_reverse[end_encoding]
                    
                    if start_slot_name != 'no_slot':
                        start_tokens_from_encoding.append(current_chunk_token)
                        start_labels_from_encoding.append(start_slot_name)
                    if end_slot_name != 'no_slot':
                        end_tokens_from_encoding.append(current_chunk_token)
                        end_labels_from_encoding.append(end_slot_name)
                        
            
                self.assertEqual(start_tokens_from_encoding, start_tokens_from_doc)
            self.assertEqual(end_tokens_from_encoding, end_tokens_from_doc)
            
            for i in range(len(tokens_from_doc)):
                print(tokens_from_doc[i], start_tokens_from_encoding[i], end_tokens_from_encoding[i])
            print('-----------------')
            
            self.assertEqual(start_labels_from_encoding, all_referencing_slot_names)
            self.assertEqual(end_labels_from_encoding, all_referencing_slot_names)
                
                
if __name__ == '__main__':
    unittest.main()