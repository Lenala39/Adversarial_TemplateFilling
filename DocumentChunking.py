from Chunk import *
from Document import *
from copy import deepcopy
        


class DocumentChunking:
    def __init__(self, max_chunk_size, bert_dimension=512):   
        # max tokens of each chunk
        self._max_chunk_size = max_chunk_size
        self._bert_dimension = bert_dimension
        # list of chunks
        self._chunks = []
        
    def get_bert_dimension(self):
        return self._bert_dimension

    def append_chunk(self, chunk):
        self._chunks.append(chunk)
    
    def split_sentence_seq_for_chunking(self, max_chunk_size):
        
        # list of lists; inner list contains Sentence objects
        sentence_blocks = []
        
        # list of sentence for current block
        current_block = []
        
        left_chunk_size = max_chunk_size - 1 # -1 since first token is [CLS]
        
        for chunk in self.get_chunks():
            # split sentences
            for sentence in chunk.get_sentences():
                num_tokens = sentence.get_num_tokens()
                
                # check if sentence fits into chunk at all
                if num_tokens + 2 > max_chunk_size:
                    raise Exception('Sentence does not fit into chunk')
                
                # does sentence fit into current chunk?
                if left_chunk_size >= num_tokens + 1: # + 1 because of [SEP] token
                    
                    # add sentence to current block
                    current_block.append(sentence)
                    
                    # decrease left chunk size by number of tokens used by current sentence
                    left_chunk_size -= (num_tokens + 1)

                # sentence does not fit into chunk; create new one
                else: 
                    
                    # save current block representing current chunk
                    sentence_blocks.append(current_block)
                    
                    # new list for current block
                    current_block = [sentence]
                    
                    # initial left chunk size of new block
                    left_chunk_size = max_chunk_size -  (num_tokens + 2) # +2 for CLS token and SEP token
                    
        # check if there is an unsaved block
        if len(current_block) > 0:
            
            # save last block
            sentence_blocks.append(current_block)

        return sentence_blocks
    
    def get_sentence_by_index(self, index):
        # TODO: write method
        for chunk in self.get_chunks():
            sentence = chunk.get_sentence_by_index(index)
            if sentence:
                return sentence
        
        return None 
            

    def set_from_chunks(self, new_chunk, sentence_index):
        old_chunks = self.get_chunks()
        self._chunks = []

        new_chunk_index = new_chunk.get_chunk_index()

        # set the new chunk object at the correct index
        for chunk in old_chunks:
            if chunk.get_chunk_index() == new_chunk_index:
                self._chunks.append(new_chunk)
            else:
                self._chunks.append(chunk)
        # in case chunk needs to be split differnetly -> done in here
        sentence_blocks = self.split_sentence_seq_for_chunking(self._max_chunk_size)

        self._chunks = []

        # create chunks from sentence blocks
        current_chunk_index = 0
        
        for sentence_block in sentence_blocks:
            chunk = Chunk(sentence_block, current_chunk_index)
            # chunk.update_sentence_offsets() # should not be necessary, done in init
            self.append_chunk(chunk)
            if sentence_index in [s.get_index() for s in sentence_block]:
                new_chunk_index = current_chunk_index
            current_chunk_index += 1
        
        return self._chunks[new_chunk_index]
        
    def set_from_document(self, document):
        self._chunks = []
        
        # split sentences of document into blocks which
        # fit into chunks
        sentence_blocks = document.split_sentence_seq_for_chunking(self.get_max_chunk_size())
        
        # create chunks from sentence blocks
        current_chunk_index = 0
        
        for sentence_block in sentence_blocks:
            chunk = Chunk(sentence_block, current_chunk_index)
            self.append_chunk(chunk)
            current_chunk_index += 1

    def set_all_chunks(self, new_chunks):
        self._chunks = []
        for c in new_chunks:
            self._chunks.append(c)
    
    def get_max_chunk_size(self):
        return self._max_chunk_size
    
    
    def get_num_chunks(self):
        return len(self._chunks)
    
    
    def get_chunks(self):
        return self._chunks
    
    def get_sentences(self):
        all_sentences = []
        for chunk in self._chunks:
            for sentence in chunk.get_sentences():
                all_sentences.extend(sentence.get_entities())

        return all_sentences
        
    def get_entities(self):
        all_entities = []
        for chunk in self._chunks:
            for sentence in chunk.get_sentences():
                all_entities.extend(sentence.get_entities())
        
        return sorted(all_entities, key=lambda entity: entity.get_global_entity_index())
    
    def get_chunk_by_index(self, chunk_index):
        chunks_dict = {chunk.get_chunk_index():chunk for chunk in self.get_chunks()}
        
        # check if chunk index is valid
        if chunk_index not in chunks_dict:
            raise IndexError('Invalid chunk index')
            
        return chunks_dict[chunk_index]
    
    def get_sentence_offsets(self, flattened=True):
        if flattened:
            sentence_offsets_all = []
            for c in self.get_chunks():
                sentence_offsets_all.extend(list(c._sentence_offsets.keys()))

        else:
            sentence_offsets_all = [list(chunk._sentence_offsets.keys()) for chunk in self.get_chunks()]
        return sentence_offsets_all
    
    def get_sentence_indices(self):
        sentence_indices = []
        for chunk in self.get_chunks():
            sentence_indices.extend(chunk.get_sentence_indices())
        
        return sentence_indices

    def set_chunk_by_index(self, chunk_index, new_chunk, sentence_index):
        chunks_dict = {chunk.get_chunk_index():chunk for chunk in self.get_chunks()}

        sentence_indeces_old = self.get_sentence_indices()

        # check if chunk index is valid
        if chunk_index not in chunks_dict:
            raise IndexError('Invalid chunk index')
        
        def check_missing_sentences():
            # new chunk has other sentence offsets than old chunk
            missing_sentences = list(set(chunks_dict[chunk_index]._sentence_offsets.keys()).symmetric_difference(set(new_chunk._sentence_offsets.keys())))
            if len(missing_sentences) > 0:
                for missing_sent_index in missing_sentences:
                    missing_sent = self._chunks[chunk_index].get_sentence_by_index(missing_sent_index)
                    
                    if not missing_sent:
                        print(f"sentence with index {missing_sent_index} not found in doc chunking (chunk index {chunk_index}")
                        if self._chunks[min(chunk_index+1, len(self._chunks)-1)].get_sentence_by_index(missing_sent_index):
                            missing_sent = self._chunks[min(chunk_index+1, len(self._chunks)-1)].get_sentence_by_index(missing_sent_index)
                            print(f"missing sent found in chunk after with chunk index {chunk_index+1}")
                        elif self._chunks[min(0,chunk_index-1)].get_sentence_by_index(missing_sent_index):
                            missing_sent = self._chunks[min(0,chunk_index-1)].get_sentence_by_index(missing_sent_index)
                            print(f"missing sent found in chunk before with chunk index {chunk_index-1}")
                        else:
                            print("Missing_sent not found!!")
                    
                    try:
                        self._chunks[chunk_index + 1].add_sentence(new_sentence=missing_sent, sentence_index=missing_sent_index)
                    except IndexError:
                        new_chunk = Chunk(sentences=[missing_sent], chunk_index=chunk_index+1)
                        self.append_chunk(new_chunk)
                    
                        
        self._chunks[chunk_index] = new_chunk

        #sentence_indeces_new = self.get_sentence_indices()

        #if sentence_indeces_new != sentence_indeces_old:    
        #    self.update_sentence_offsets()

    def update_sentence_offsets(self):
        for chunk in self.get_chunks():
            chunk.update_sentence_offsets()


    def append_chunk(self, chunk):
        self._chunks.append(chunk)
        
        
    def set_entity_chunk_indices(self, entities):
        for entity in entities:
            chunk_found = False
            entity_sentence_index = entity.get_sentence_index()
            
            for chunk in self.get_chunks():
                if entity_sentence_index in chunk.get_sentence_indices():
                    chunk_found = True
                    entity.set_chunk_index(chunk.get_chunk_index())
                    break
                
            # check if a chunk for current entity was found
            if not chunk_found:
                raise Exception('ERROR: no chunk found for entity')
                
                
                
    def get_entity_start_end_indices(self, entities):
        # lists of [chunk_index, in_chunk_offset] lists representing indices
        # into doc chunking of entity start/end positions
        entity_start_indices = []
        entity_end_indices = []
        
        chunk_indices = {chunk:chunk.get_sentence_indices() for chunk in self.get_chunks()}

        for entity in entities:
            chunk_index = None
            entity_sentence_index = entity.get_sentence_index()
            
            # get chunk containing entity ################################
            for chunk, sentence_indices in chunk_indices.items():
                if entity_sentence_index in sentence_indices:
                    chunk_index = chunk.get_chunk_index()
                    break
                
            # check if a chunk for current entity was found
            if chunk_index is None:
                raise Exception('ERROR: no chunk found for entity')
                
            # compute start/end positions of entity in chunk
            in_chunk_sentence_offset = chunk.get_sentence_offset(entity_sentence_index)
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
            for chunk in self.get_chunks():
                all_entities.extend(chunk.get_entities())
            self._entities = all_entities
        
            return self._entities
    
    def __copy__(self):
        newone = type(self)(self._max_chunk_size)
        newone.__dict__.update(self.__dict__)
        return newone

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

