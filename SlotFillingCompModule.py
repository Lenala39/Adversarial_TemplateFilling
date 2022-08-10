import dis
from numpy import argmax
import tensorflow as tf
import tensorflow_hub as hub
from official.nlp.bert import tokenization
from official.nlp import optimization
from tqdm import tqdm
import json 

from ctro import *
from Document import *
from DocumentChunking import *
from DocumentEncoder import *
from EntityCompatibilityCollection import *
from SlotFillingModel import *
from EntityDecoder import *
from EntityAligner import *
from group_optimizer import *
from functions import *
from stats_printer import *
from helper_functions import __get_file_descr_with_params__, save_modified_data, parse_cli_arguments
from DataImporter import DataImporter
import pandas as pd 

MIN_SLOT_FREQ = 1
MAX_CHUNK_SIZE = 400
BERT_MODEL_DIM = 768

EPOCHS = 30

# model_prefix = 'glaucoma'
dump_file_path = 'glaucoma_slot_filling.dump'
bert_model_name = "https://tfhub.dev/google/experts/bert/pubmed/2"


# train step signature
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # token ids
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # token masks
    # entity start_pos indices
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    # entity end pos indices
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    # entity1 of entity compatibility pairs
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    # entity2 of entity compatibility pairs
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    # gt compatibility of entity pairs
    tf.TensorSpec(shape=(None,), dtype=tf.float32),
    # entity start positions
    tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    # entity end positions
    tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
]

forward_pass_positions_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # token ids
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # sentence tokens
]

forward_pass_encoder_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # token ids
    tf.TensorSpec(shape=(None, None), dtype=tf.int32)  # token masks
]

tf_compute_entity_representations_signature = [
    tf.TensorSpec(shape=(None, None, None),
                  dtype=tf.float32),  # encoder output
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # start token indices
    tf.TensorSpec(shape=(None, None), dtype=tf.int32)  # end token indices
]

tf_compute_entity_compatibilities_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # all entity vectors
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # entity1 indices
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # entity2 indices
]

# train step signature
test_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # token ids
tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # sentence tokens
    # entity start positions
    tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    # entity end positions
    tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
]

# create bert tokenizer
FullTokenizer = tokenization.FullTokenizer
bert_layer = hub.KerasLayer(bert_model_name, trainable=False)
# The vocab file of bert for tokenizer
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
tokenizer = FullTokenizer(vocab_file)


def create_document_chunkings(documents, max_chunk_size):
    document_chunkings = []

    for doc in documents:
        doc_chunking = DocumentChunking(max_chunk_size)
        doc_chunking.set_from_document(doc)
        document_chunkings.append(doc_chunking)

    return document_chunkings


def create_batches_from_chunking(document_chunkings, document_encoder, slot_indices):
    batches = []

    for i in range(len(document_chunkings)):
        
        doc_chunking = document_chunkings[i]
        batch = dict()
        num_slot_labels = len(slot_indices)

        # tokens ####################################
        batch['token_ids'] = tf.convert_to_tensor(document_encoder.encode_tokens_document_chunking(doc_chunking), tf.int32)
        batch['token_masks'] = tf.convert_to_tensor(document_encoder.create_token_masks_document_chunking(doc_chunking), tf.int32)

        # indices into doc chunking of entity start/end tokens #################
        entities = doc_chunking.get_entities()
        entity_start_token_indices, entity_end_token_indices = doc_chunking.get_entity_start_end_indices(entities)
        batch['entity_start_token_indices'] = tf.convert_to_tensor(entity_start_token_indices, tf.int32)
        batch['entity_end_token_indices'] = tf.convert_to_tensor(entity_end_token_indices, tf.int32)

        # entity pair compatibility ###############################
        entity1_indices_all = []
        entity2_indices_all = []
        entity_compatibilities_all = []

        for template_type in ontology.used_group_names:
            template_type_entities = select_entities_by_template_type(
                entities, template_type) # all entities with in that group with labels like Journal/Title/ ... for template_type Publication
            if len(template_type_entities) < 2:  # no pairs
                continue

            # compatibility
            entity_comp_collection = EntityCompatibilityCollection(
                template_type_entities)
            entity_compatibilities_all.extend(
                entity_comp_collection.get_ground_truth_compatibility_list())

            # entity indices
            entity1_indices, entity2_indices = entity_comp_collection.get_entity_indices_of_pairs()
            entity1_indices_all.extend(entity1_indices)
            entity2_indices_all.extend(entity2_indices)

        # update batch dirct
        entity1_indices_all = tf.convert_to_tensor(entity1_indices_all, dtype=tf.int32)
        entity2_indices_all = tf.convert_to_tensor(entity2_indices_all, dtype=tf.int32)

        entity1_indices_all = tf.expand_dims(entity1_indices_all, axis=-1) #put each entry in seperate vec: shape [234] -> [234,1]
        entity2_indices_all = tf.expand_dims(entity2_indices_all, axis=-1)

        batch['entity_compatibilities'] = tf.convert_to_tensor(entity_compatibilities_all, tf.float32)
        batch['entity1_compatibility_pairs'] = tf.convert_to_tensor(entity1_indices_all, tf.int32)
        batch['entity2_compatibility_pairs'] = tf.convert_to_tensor(entity2_indices_all, tf.int32)

        # entity start/end positions ################################
        encoded_start_positions, encoded_end_positions = document_encoder.encode_entity_positions_document_chunking(doc_chunking)
        batch['entity_start_positions'] = tf.one_hot(encoded_start_positions, num_slot_labels)
        batch['entity_end_positions'] = tf.one_hot(encoded_end_positions, num_slot_labels)

        batches.append(batch)

    return batches


def create_batches(documents, document_chunkings, document_encoder, slot_indices):
    batches = []

    for i in range(len(documents)):
        document = documents[i]
        doc_chunking = document_chunkings[i]
        batch = dict()
        num_slot_labels = len(slot_indices)

        # tokens ####################################
        batch['token_ids'] = tf.convert_to_tensor(document_encoder.encode_tokens_document_chunking(doc_chunking), tf.int32)
        batch['token_masks'] = tf.convert_to_tensor(document_encoder.create_token_masks_document_chunking(doc_chunking), tf.int32)

        # indices into doc chunking of entity start/end tokens #################
        entities = document.get_entities()
        entity_start_token_indices, entity_end_token_indices = doc_chunking.get_entity_start_end_indices(entities)
        batch['entity_start_token_indices'] = tf.convert_to_tensor(entity_start_token_indices, tf.int32)
        batch['entity_end_token_indices'] = tf.convert_to_tensor(entity_end_token_indices, tf.int32)

        # entity pair compatibility ###############################
        entity1_indices_all = []
        entity2_indices_all = []
        entity_compatibilities_all = []

        for template_type in ontology.used_group_names:
            template_type_entities = select_entities_by_template_type(
                entities, template_type) # all entities with in that group with labels like Journal/Title/ ... for template_type Publication
            if len(template_type_entities) < 2:  # no pairs
                continue

            # compatibility
            entity_comp_collection = EntityCompatibilityCollection(
                template_type_entities)
            entity_compatibilities_all.extend(
                entity_comp_collection.get_ground_truth_compatibility_list())

            # entity indices
            entity1_indices, entity2_indices = entity_comp_collection.get_entity_indices_of_pairs()
            entity1_indices_all.extend(entity1_indices)
            entity2_indices_all.extend(entity2_indices)

        # update batch dirct
        entity1_indices_all = tf.convert_to_tensor(entity1_indices_all, dtype=tf.int32)
        entity2_indices_all = tf.convert_to_tensor(entity2_indices_all, dtype=tf.int32)

        entity1_indices_all = tf.expand_dims(entity1_indices_all, axis=-1) #put each entry in seperate vec: shape [234] -> [234,1]
        entity2_indices_all = tf.expand_dims(entity2_indices_all, axis=-1)

        batch['entity_compatibilities'] = tf.convert_to_tensor(entity_compatibilities_all, tf.float32)
        batch['entity1_compatibility_pairs'] = tf.convert_to_tensor(entity1_indices_all, tf.int32)
        batch['entity2_compatibility_pairs'] = tf.convert_to_tensor(entity2_indices_all, tf.int32)

        # entity start/end positions ################################
        encoded_start_positions, encoded_end_positions = document_encoder.encode_entity_positions_document_chunking(doc_chunking)
        batch['entity_start_positions'] = tf.one_hot(encoded_start_positions, num_slot_labels)
        batch['entity_end_positions'] = tf.one_hot(encoded_end_positions, num_slot_labels)

        batches.append(batch)

    return batches


class SlotFillingCompModule:

    def __init__(self, tokenizer):
        self.data_importer = DataImporter(tokenizer)

    def count_slots(self):
        slot_counts = dict()

        for document in self.documents_train:
            for sentence in document.get_sentences():
                for entity in sentence.get_entities():

                    for slot_name in list(entity.get_referencing_slot_names()):
                        if slot_name == 'type':
                            continue

                        if slot_name in slot_counts:
                            slot_counts[slot_name] += 1
                        else:
                            slot_counts[slot_name] = 0
        
        for document in self.documents_test:
            for sentence in document.get_sentences():
                for entity in sentence.get_entities():

                    for slot_name in list(entity.get_referencing_slot_names()):
                        if slot_name == 'type':
                            continue

                        if slot_name in slot_counts:
                            slot_counts[slot_name] += 1
                        else:
                            slot_counts[slot_name] = 0

        return slot_counts

    def create_slot_indices(self, min_slot_freq):
        slot_counts = self.count_slots()

        slot_indices = dict()
        slot_indices_reverse = dict()

        # index of 'no_slot' label
        slot_indices['no_slot'] = 0
        slot_indices_reverse[0] = 'no_slot'

        # 'real' slot labels
        for slot_name in slot_counts:

            if slot_counts[slot_name] >= min_slot_freq:
                slot_index = len(slot_indices)

                slot_indices[slot_name] = len(slot_indices)
                slot_indices_reverse[slot_index] = slot_name

        # save dicts as object variables
        self.slot_indices = slot_indices
        self.slot_indices_reverse = slot_indices_reverse
        self.used_slots = list(slot_indices.keys())

        return slot_indices, slot_indices_reverse

    def create_document_chunkings(self, max_chunk_size):
        self.document_chunkings_train = create_document_chunkings(self.documents_train, max_chunk_size)
        self.document_chunkings_validation = create_document_chunkings(self.documents_validation, max_chunk_size)
        self.document_chunkings_test = create_document_chunkings(self.documents_test, max_chunk_size)

    def create_batches(self):
        # create document encoder
        document_encoder = DocumentEncoder(tokenizer, self.slot_indices, 'no_slot')

        # create batches
        self.batches_train = create_batches(self.documents_train, self.document_chunkings_train, document_encoder, self.slot_indices)
        self.batches_validation = create_batches(self.documents_validation, self.document_chunkings_validation, document_encoder, self.slot_indices)
        self.batches_test = create_batches(self.documents_test, self.document_chunkings_test, document_encoder, self.slot_indices)

    def create_optimizer(self):
        steps_per_epoch = len(self.batches_train)
        num_train_steps = steps_per_epoch * EPOCHS
        num_warmup_steps = int(0.1*num_train_steps)

        init_lr = 3e-5
        self.optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                       num_train_steps=num_train_steps,
                                                       num_warmup_steps=num_warmup_steps,
                                                       optimizer_type='adamw')

    def create_model(self):
        print(f"Creating model with {BERT_MODEL_DIM} elements for BERT and {len(self.slot_indices)} one-hot vecs")
        self.slot_filling_model = SlotFillingModel(bert_model_name, BERT_MODEL_DIM, len(self.slot_indices))

    def prepare(self, disease_str, optimizer=False, load_data=False, slot_index_path=None):
        if load_data:
            print("Creating doc chunkings")
            self.create_document_chunkings(MAX_CHUNK_SIZE)
            print("Creating batches")
            self.create_batches()

        print("Creating model")
        self.create_model()

        if optimizer:
            print("Creating optimizer")
            self.create_optimizer()

        # keras metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        print("PREPARE >> DONE")

    def call_bert_layer(self, token_ids, token_ids_masks, training=False):
        input_dict = {}
        input_dict['input_word_ids'] = token_ids
        input_dict['input_mask'] = token_ids_masks
        input_dict['input_type_ids'] = tf.zeros_like(token_ids, dtype=tf.int32)

        return self.slot_filling_model.bert_layer(input_dict, training=training)["sequence_output"]

    @tf.function(input_signature=train_step_signature)
    def train_step(self, token_ids, token_id_masks,
                   entity_start_token_indices, entity_end_token_indices,
                   entity1_compatibility_pairs, entity2_compatibility_pairs, gt_entity_compatibilities,
                   entity_start_position_labels, entity_end_position_labels):

        with tf.GradientTape() as tape:
            # call BERT encoder
            token_embeddings = self.call_bert_layer(token_ids, token_id_masks, training=True)

            # compute entity representation
            entity_vectors = self.slot_filling_model.compute_entity_representation(token_embeddings, entity_start_token_indices,
                                                                                   entity_end_token_indices, training=True)

            # loss entity start/end positions ############################################################
            start_position_logits = self.slot_filling_model.dense_entity_start_positions(token_embeddings)
            end_position_logits = self.slot_filling_model.dense_entity_end_positions(token_embeddings)

            loss_entity_start_positions = tf.nn.softmax_cross_entropy_with_logits(labels=entity_start_position_labels, logits=start_position_logits)
            loss_entity_end_positions = tf.nn.softmax_cross_entropy_with_logits(labels=entity_end_position_labels, logits=end_position_logits)

            # mask invalid positions
            bool_token_id_masks = tf.greater(token_id_masks, 0)
            num_tokens = tf.reduce_sum(token_id_masks)
            num_tokens = tf.cast(num_tokens, tf.float32)

            loss_entity_start_positions = tf.boolean_mask(
                loss_entity_start_positions, bool_token_id_masks)
            loss_entity_end_positions = tf.boolean_mask(
                loss_entity_end_positions, bool_token_id_masks)

            loss_entity_start_positions = tf.reduce_sum(
                loss_entity_start_positions) / num_tokens
            loss_entity_end_positions = tf.reduce_sum(
                loss_entity_end_positions) / num_tokens

            # loss entity compatibility ##############################################################
            entity1_vectors = tf.gather_nd(entity_vectors, entity1_compatibility_pairs)
            entity2_vectors = tf.gather_nd(entity_vectors, entity2_compatibility_pairs)
            entity_pair_representation = entity1_vectors + entity2_vectors
            entity_pair_representation = tf.reshape(entity_pair_representation, shape=(-1, self.slot_filling_model.bert_model_dim))

            entity_pair_compatibility_logits = self.slot_filling_model.dense_entity_compatibility(entity_pair_representation)
            entity_pair_compatibility_logits = tf.squeeze(entity_pair_compatibility_logits, axis=-1)

            entity_compatibility_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=entity_pair_compatibility_logits,
                                                                                labels=gt_entity_compatibilities)
            entity_compatibility_loss = tf.reduce_mean(entity_compatibility_loss)

            # total loss
            loss = loss_entity_start_positions + loss_entity_end_positions + entity_compatibility_loss

        # gradient descent
        trainable_variables = self.slot_filling_model.trainable_weights
        gradients = tape.gradient(loss, trainable_variables)
        #gradients = [tf.clip_by_norm(gradient, 1.0) for gradient in gradients]
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        # update metrics
        self.train_loss(loss)

    def train(self, num_epochs, model_prefix, model_descr=None):
        for i in range(num_epochs):
            for batch in self.batches_train:

                self.train_step(batch['token_ids'],
                                batch['token_masks'],
                                batch['entity_start_token_indices'],
                                batch['entity_end_token_indices'],
                                batch['entity1_compatibility_pairs'],
                                batch['entity2_compatibility_pairs'],
                                batch['entity_compatibilities'],
                                batch['entity_start_positions'],
                                batch['entity_end_positions'])

            # print loss after each epoch
            print('loss: ' + str(self.train_loss.result()))
            self.train_loss.reset_states()
            
            # get path 
            dir_path = os.path.dirname(os.path.realpath(__file__))
            if (i+1) % 10 == 0 and i+1 >= 30:
                if model_descr:
                    model_filename = f"{model_prefix}_{model_descr}_{str(i+1)}epochs"
                else:
                    model_filename = f"{model_prefix}_{str(i+1)}epochs"

                full_filename = os.path.join(dir_path, "slot_filling_models/", f"{disease_str}/", model_filename)
                print(f"Saving model after {i} epochs to {full_filename}")
                self.slot_filling_model.save_weights(full_filename)
                # self.slot_filling_model.save(full_filename)

    def save_model_weights(self, filename):
        self.slot_filling_model.save_weights(filename)

    def load_model_weights(self, filename):
        self.slot_filling_model.load_weights(filename)

    @tf.function(input_signature=test_step_signature)
    def test_step(self, token_ids, token_id_masks):
        token_embeddings = self.call_bert_layer(token_ids, token_id_masks, training=False)
        
        start_position_logits = self.slot_filling_model.dense_entity_start_positions(token_embeddings)
        end_position_logits = self.slot_filling_model.dense_entity_end_positions(token_embeddings)

        return start_position_logits, end_position_logits#, loss

    def decode_entities(self, document_index):
        entity_decoder = EntityDecoder(self.slot_indices, 'no_slot')
        doc_chunking = self.document_chunkings_test[document_index]
        batch = self.batches_test[document_index]
        
        return entity_decoder.decode_document_chunking(batch['entity_start_positions'],
                                                       batch['entity_end_positions'], 
                                                       doc_chunking)

    def call_encoder_test_set(self):
        self.encoder_output_test_set = []

        for test_batch in self.batches_test:
            self.encoder_output_test_set.append(self.forward_pass_encoder(test_batch['token_ids'],
                                                                          test_batch['token_masks']))

    @tf.function(input_signature=tf_compute_entity_representations_signature)
    def tf_compute_entity_representations(self, encoder_output, entity_start_token_indices, entity_end_token_indices):
        return self.slot_filling_model.compute_entity_representation(encoder_output,
                                                                     entity_start_token_indices,
                                                                     entity_end_token_indices)

    @tf.function(input_signature=tf_compute_entity_compatibilities_signature)
    def tf_compute_entity_compatibilities(self, all_entity_vectors, entity1_indices, entity2_indices):
        # gather all entity representations with an index in entityX_indices
        entity1_vectors = tf.gather_nd(all_entity_vectors, entity1_indices) 
        entity2_vectors = tf.gather_nd(all_entity_vectors, entity2_indices)

        entity_pair_representation = entity1_vectors + entity2_vectors
        entity_pair_representation = tf.reshape(entity_pair_representation,
                                                shape=(-1, self.slot_filling_model.bert_model_dim))

        entity_pair_combatibilities = self.slot_filling_model.dense_entity_compatibility(entity_pair_representation)
        entity_pair_combatibilities = tf.squeeze(entity_pair_combatibilities, axis=-1)
        
        return entity_pair_combatibilities

    def create_entity_compatibility_collection(self, all_entity_vectors, entities, sigmoid=True):
        entity_compatibility_collection = EntityCompatibilityCollection(entities)
        entity1_indices, entity2_indices = entity_compatibility_collection.get_entity_indices_of_pairs()

        entity1_indices = tf.convert_to_tensor(entity1_indices, tf.int32)
        entity2_indices = tf.convert_to_tensor(entity2_indices, tf.int32)

        entity1_indices = tf.expand_dims(entity1_indices, axis=-1)
        entity2_indices = tf.expand_dims(entity2_indices, axis=-1)

        entity_compatibilities = self.tf_compute_entity_compatibilities(all_entity_vectors,
                                                                        entity1_indices,
                                                                        entity2_indices).numpy().tolist()
        
        entity_compatibility_collection.set_entity_pair_compatibilities_no_sigmoid(entity_compatibilities)
        # sigmoid
        entity_combatibilities = tf.math.sigmoid(entity_compatibilities).numpy().tolist()
        entity_compatibility_collection.set_entity_pair_compatibilities(entity_combatibilities)
                                                                       
        return entity_compatibility_collection

    @tf.function(input_signature=forward_pass_encoder_signature)
    def forward_pass_encoder(self, token_ids, token_id_masks):
        return self.call_bert_layer(token_ids, token_id_masks, training=False)

    @tf.function(input_signature=forward_pass_positions_signature)
    def forward_pass_entity_positions(self, token_ids, token_id_masks):
        token_embeddings = self.call_bert_layer(token_ids, token_id_masks, training=False)

        entity_start_positions = self.slot_filling_model.dense_entity_start_positions(token_embeddings)
        entity_end_positions = self.slot_filling_model.dense_entity_end_positions(token_embeddings)

        return entity_start_positions, entity_end_positions

    def get_start_end_positions(self, batches):
        
        new_entity_start_positions = []
        new_entity_end_positions = []

        for i in range(len(batches)):
            batch = batches[i]
            
            # predict entity positions
            entity_start_positions, entity_end_positions = self.forward_pass_entity_positions(batch['token_ids'],
                                                                                              batch['token_masks'])

            #entity_start_positions = tf.argmax(entity_start_positions, axis=-1)
            #entity_end_positions = tf.argmax(entity_end_positions, axis=-1)

            new_entity_start_positions.append(entity_start_positions)
            new_entity_end_positions.append(entity_end_positions)
        
        return tf.convert_to_tensor(new_entity_start_positions), tf.convert_to_tensor(new_entity_end_positions) 
           

    def evaluate_entity_prediction(self, filename=None):
        entity_decoder = EntityDecoder(self.slot_indices, 'no_slot')
        entity_aligner = EntityAligner()
        stats_dict = dict()
        
        for i in tqdm(range(len(self.documents_test))):
            batch = self.batches_test[i]
            if isinstance(batch, list):
                batch = batch[0]
            document = self.documents_test[i]
            doc_chunking = self.document_chunkings_test[i]

            # predict entity positions
            entity_start_positions, entity_end_positions = self.forward_pass_entity_positions(batch['token_ids'],
                                                                                              batch['token_masks'])

            entity_start_positions = tf.argmax(entity_start_positions, axis=-1)
            entity_end_positions = tf.argmax(entity_end_positions, axis=-1)

            # get gt slot referenced entities
            gt_entities = [entity for entity in document.get_entities() if len(entity.get_referencing_slot_names()) > 0]

            # predict entities
            predicted_entities = entity_decoder.decode_document_chunking(entity_start_positions,
                                                                         entity_end_positions,
                                                                         doc_chunking)

            aligned_entity_pairs = entity_aligner.align_entities_exact(gt_entities, predicted_entities)
            entity_aligner.update_stats_dict(stats_dict, aligned_entity_pairs)

        print_event_extraction_stats(stats_dict, self.used_slots, filename=filename)

    def compute_entities_from_input(self, batch, doc_chunking, document):
        entity_decoder = EntityDecoder(self.slot_indices, 'no_slot')
        entity_start_positions, entity_end_positions = self.forward_pass_entity_positions(batch['token_ids'],
                                                                                              batch['token_masks'])

        entity_start_positions = tf.argmax(entity_start_positions, axis=-1)
        entity_end_positions = tf.argmax(entity_end_positions, axis=-1)

        # predict entities
        entities = entity_decoder.decode_document_chunking(entity_start_positions, entity_end_positions, doc_chunking)
        # assign indices to entities    
        for i, entity in enumerate(entities):
            entity.set_global_entity_index(i)
        # set entity tokens
        document.set_entity_tokens(entities)
        return entities

    def entities_changed(self, original_entities, modified_entities):

        if len(original_entities) != len(modified_entities):
            return 1

        original_entities = set(original_entities)
        modified_entities = set(modified_entities)

        if original_entities == modified_entities:
            return 0

        # overlap
        same_entities = list(original_entities.intersection(modified_entities))
        # what was not found
        missed_entities = list(original_entities.difference(modified_entities))
        # found additionally (wrongly?)
        newly_added_entities = list(modified_entities.difference(original_entities))

        if len(missed_entities) > 0 or len(newly_added_entities) > 0:
            return 1

    def compute_entity_comp_for_attacking(self, document, doc_chunking, batches, sigmoid=True):
        entity_decoder = EntityDecoder(self.slot_indices, 'no_slot')
        group_collection = document.get_abstract().group_collection

        batches_comp_scores = []
        batches_comp_scores_no_sigmoid = []

        for batch in batches:
        
            encoder_output = self.forward_pass_encoder(batch['token_ids'], batch['token_masks'])


            # predict entities #########################################
            entity_start_positions, entity_end_positions = self.forward_pass_entity_positions(batch['token_ids'], batch['token_masks'])

            entity_start_positions = tf.argmax(entity_start_positions, axis=-1)
            entity_end_positions = tf.argmax(entity_end_positions, axis=-1)

            # predict entities
            entities = entity_decoder.decode_document_chunking(entity_start_positions, entity_end_positions, doc_chunking)
            
            # assign indices to entities
            for i, entity in enumerate(entities):
                entity.set_global_entity_index(i)
            
            # set entity tokens
            document.set_entity_tokens(entities)

            # compute entity representations
            entity_start_token_indices, entity_end_token_indices = doc_chunking.get_entity_start_end_indices(entities)

            entity_vectors = self.tf_compute_entity_representations(encoder_output,
                                                                    entity_start_token_indices,
                                                                    entity_end_token_indices)

            comp_values_no_sigmoid_all = []
            comp_values_all = []
            for template_type in group_collection.groups:
                template_type_entities = select_entities_by_template_type(entities, template_type)
                gt_templates = group_collection.groups[template_type]
                num_template_instances = len(gt_templates)

                if num_template_instances == 0:
                    continue

                if len(template_type_entities) > 1 and num_template_instances > 1:
                    entity_compatibility_collection = self.create_entity_compatibility_collection(entity_vectors,
                                                                                                    template_type_entities, sigmoid=sigmoid)
                else:
                    entity_compatibility_collection = EntityCompatibilityCollection(template_type_entities)

                comp_values = entity_compatibility_collection.get_entity_pair_compatibilities()
                comp_values_all.append(comp_values)
                comp_values_no_sigmoid = entity_compatibility_collection.get_entity_pair_compatibilities_no_sigmoid()
                comp_values_no_sigmoid_all.append(comp_values_no_sigmoid)

            batches_comp_scores.append(comp_values_all)
            batches_comp_scores_no_sigmoid.append(comp_values_no_sigmoid_all)
        
        return batches_comp_scores, batches_comp_scores_no_sigmoid

    def evaluate_slot_filling(self, beam_size, sigmoid=True, filename=None):
        random_stats = F1StatisticsCollection()
        intra_compatibility_stats = F1StatisticsCollection()
        entity_decoder = EntityDecoder(self.slot_indices, 'no_slot')

        for test_doc_index in tqdm(range(len(self.documents_test))):
            encoder_output = self.encoder_output_test_set[test_doc_index] # forward_pass_encoder
            document = self.documents_test[test_doc_index]
            group_collection = document.get_abstract().group_collection
            doc_chunking = self.document_chunkings_test[test_doc_index]
            batch = self.batches_test[test_doc_index]

            #entities = document.get_entities()
            # predict entities #########################################
            entity_start_positions, entity_end_positions = self.forward_pass_entity_positions(batch['token_ids'],
                                                                                              batch['token_masks'])

            entity_start_positions = tf.argmax(entity_start_positions, axis=-1)
            entity_end_positions = tf.argmax(entity_end_positions, axis=-1)

            # predict entities
            entities = entity_decoder.decode_document_chunking(entity_start_positions,
                                                                entity_end_positions,
                                                                doc_chunking)

            # assign indices to entities
            for i, entity in enumerate(entities):
                entity.set_global_entity_index(i)

            # set entity tokens
            document.set_entity_tokens(entities)

            # compute entity representations
            entity_start_token_indices, entity_end_token_indices = doc_chunking.get_entity_start_end_indices(entities)
            entity_vectors = self.tf_compute_entity_representations(encoder_output,
                                                                    entity_start_token_indices,
                                                                    entity_end_token_indices)

            
            for template_type in group_collection.groups:
                
                template_type_entities = select_entities_by_template_type(entities, template_type)
                gt_templates = group_collection.groups[template_type]
                num_template_instances = len(gt_templates)

                if num_template_instances == 0:
                    continue

                if len(template_type_entities) > 1 and num_template_instances > 1:
                    entity_compatibility_collection = self.create_entity_compatibility_collection(entity_vectors,
                                                                                                  template_type_entities, 
                                                                                                  sigmoid=sigmoid)
                else:
                    entity_compatibility_collection = EntityCompatibilityCollection(template_type_entities)

                # optimize entity assignment
                configuration_optimizer = ConfigurationOptimizer(num_template_instances, entity_compatibility_collection)

                # estimate configurations for different settings
                random_configuration = configuration_optimizer.create_random_configuration()
                best_intra_configuration = configuration_optimizer.beam_search(beam_size)

                # update random stats ##############################################
                predicted_templates = random_configuration.create_groups()
                aligned_groups = align_groups(gt_templates, predicted_templates, self.used_slots)

                for gt_group, predicted_group in aligned_groups:
                    random_stats.update(gt_group, predicted_group, self.used_slots)

                # update intra compatibility stats #########################################
                predicted_templates = best_intra_configuration.create_groups()
                aligned_groups = align_groups(gt_templates, predicted_templates, self.used_slots)

                for gt_group, predicted_group in aligned_groups:
                    intra_compatibility_stats.update(gt_group, predicted_group, self.used_slots)

        print_slot_prediction_stats([[random_stats], [intra_compatibility_stats], [intra_compatibility_stats]], filename=filename)

    def return_mixed_attacked_doc_chunking(self, abstract_id, dc_index, sentence_boundaries, summary_of_attacks, augment_filename):
        best_attacked_sentence_ids = self.return_best_attack_per_chunk(abstract_id=abstract_id, 
                                                                    sentence_boundaries=sentence_boundaries, 
                                                                    summary_of_attacks=summary_of_attacks)
        
        
        if not best_attacked_sentence_ids:
            with open(augment_filename, "a+") as augment_file:
                csv_writer = csv.writer(augment_file)
                csv_writer.writerow([abstract_id, None, None, None])

            return None 

        if "test" in augment_filename:
            document_chunkings = self.document_chunkings_test
            document_chunkings_attacked = self.document_chunkings_test_attacked
        elif "train" in augment_filename:
            document_chunkings = self.document_chunkings_train
            document_chunkings_attacked = self.document_chunkings_train_attacked

        try:
            attacked_dc = document_chunkings_attacked[dc_index]
        except IndexError:
            print(abstract_id)
            print(dc_index)

        max_len = max([len(elem) for elem in best_attacked_sentence_ids])

        for atk in best_attacked_sentence_ids:
            if len(atk) < max_len:
                pad_list = [None] * (max_len - len(atk))
                atk.extend(pad_list)
                
        if len(best_attacked_sentence_ids) == 1:
            best_attacked_sentence_ids = list(zip(best_attacked_sentence_ids[0]))
        elif len(best_attacked_sentence_ids) == 2:
            best_attacked_sentence_ids = list(zip(best_attacked_sentence_ids[0], best_attacked_sentence_ids[1]))
        elif len(best_attacked_sentence_ids) == 3:
            best_attacked_sentence_ids = list(zip(best_attacked_sentence_ids[0], best_attacked_sentence_ids[1], best_attacked_sentence_ids[2]))
        elif len(best_attacked_sentence_ids) == 4:
            best_attacked_sentence_ids = list(zip(best_attacked_sentence_ids[0], best_attacked_sentence_ids[1], best_attacked_sentence_ids[2], best_attacked_sentence_ids[3]))
        elif len(best_attacked_sentence_ids) == 5:
            best_attacked_sentence_ids = list(zip(best_attacked_sentence_ids[0], best_attacked_sentence_ids[1], best_attacked_sentence_ids[2], best_attacked_sentence_ids[3], best_attacked_sentence_ids[4]))
        elif len(best_attacked_sentence_ids) == 6:
            best_attacked_sentence_ids = list(zip(best_attacked_sentence_ids[0], best_attacked_sentence_ids[1], best_attacked_sentence_ids[2], best_attacked_sentence_ids[3], best_attacked_sentence_ids[4], best_attacked_sentence_ids[5]))


        output_csv = []
        new_doc_chunkings = []
        for atk_idx in best_attacked_sentence_ids:
            output_csv_temp = [abstract_id]
            new_dc = document_chunkings[dc_index].__deepcopy__(memo={})
            for chunk_index, chunk in enumerate(new_dc.get_chunks()):
                s_index = atk_idx[chunk_index]
                output_csv_temp.append(s_index)
                if s_index:
                    atk_sentence = attacked_dc.get_chunks()[chunk_index].get_sentence_by_index(s_index)
                    chunk.set_sentence_by_index(sentence_index=s_index, new_sentence=atk_sentence)
                    chunk.update_sentence_offsets()
                    new_dc.set_chunk_by_index(chunk_index=chunk_index, new_chunk=chunk, sentence_index=s_index)
                    
            new_dc.update_sentence_offsets()
            new_doc_chunkings.append(new_dc)
            output_csv.append(output_csv_temp)

        with open(augment_filename, "a+") as augment_file:
            csv_writer = csv.writer(augment_file)
            for elem in output_csv:
                csv_writer.writerow(elem)

        return new_doc_chunkings

    def augment_dataset(self, attack_log_path, test_or_train="train", append=True):
        document_encoder = DocumentEncoder(tokenizer, self.slot_indices, 'no_slot')

        summary_of_attacks = pd.read_csv(attack_log_path)
        attacked_dcs = []
        attacked_batches = []
        attacked_docs = [] # keep originals 

        # file to store which sentences were augmented
        only_filename_attack_log = attack_log_path.split("/")[-1]
        path = "/".join(attack_log_path.split("/")[:-1])
        augment_filename = f"{path}/AUGMENTATION_{only_filename_attack_log}"

        with open(augment_filename, "w") as augment_file:
            csv_writer = csv.writer(augment_file)
            csv_writer.writerow(["abstract_id", "chunk1", "chunk2", "chunk3"])

        if test_or_train.lower() == "test":
            document_chunkings = self.document_chunkings_test
            documents = self.documents_test
            batches = self.batches_test
        elif test_or_train.lower() == "train":
            document_chunkings = self.document_chunkings_train
            documents = self.documents_train
            batches = self.batches_train
        else:
            exit("Need to supply test or train for mode")

        for dc_index, dc in enumerate(document_chunkings):
            a_id = int(documents[dc_index]._abstract.abstract_id)

            sentence_boundaries = [c.get_sentences()[-1].get_index() for c in dc.get_chunks()]
            if len(sentence_boundaries) > 1:
                del sentence_boundaries[-1]

            attacked_doc_chunks = self.return_mixed_attacked_doc_chunking(abstract_id=a_id, dc_index=dc_index,
                                                                            sentence_boundaries=sentence_boundaries, 
                                                                            summary_of_attacks=summary_of_attacks, 
                                                                            augment_filename=augment_filename)

            if not attacked_doc_chunks:
                # append original to attacked if no attack
                attacked_batches.append(batches[dc_index])
                attacked_docs.append(documents[dc_index])
                attacked_dcs.append(document_chunkings[dc_index])
                continue 

            # recompute batch
            attacked_bs = create_batches_from_chunking(attacked_doc_chunks, document_encoder, self.slot_indices)
            attacked_dcs.extend(attacked_doc_chunks)
            attacked_batches.extend(attacked_bs)
            attacked_docs.extend([documents[dc_index]] * len(attacked_doc_chunks))

        # append to correct objects
        if append:
            if test_or_train.lower() == "test":
                self.document_chunkings_test.extend(attacked_dcs)
                self.batches_test.extend(attacked_batches)
                self.documents_test.extend(attacked_docs)

            elif test_or_train.lower() == "train":
                self.document_chunkings_train.extend(attacked_dcs)
                self.batches_train.extend(attacked_batches)
                self.documents_train.extend(attacked_docs)
        # set to only augmented
        else:
            if test_or_train.lower() == "test":
                self.document_chunkings_test = attacked_dcs
                self.batches_test = attacked_batches
                self.documents_test = attacked_docs

            elif test_or_train.lower() == "train":
                self.document_chunkings_train = attacked_dcs
                self.batches_train = attacked_batches
                self.documents_train = attacked_docs

    def return_best_attack_per_chunk(self, abstract_id, sentence_boundaries, summary_of_attacks, percentage_attacks=0.2):
        
        data = summary_of_attacks[summary_of_attacks["abstract_id"] == int(abstract_id)]
        chunks = []
        for i in range(len(sentence_boundaries)):
            bound = sentence_boundaries[i]
            if i == 0:
                temp_chunk = data[data["sentence_index"] <= bound]        
                chunks.append(temp_chunk)
            elif i != 0:
                temp_chunk = data[(data["sentence_index"] <= bound)] # lower than bound
                temp_chunk = temp_chunk[temp_chunk["sentence_index"] > sentence_boundaries[i-1]] # larger than previous chunk
                chunks.append(temp_chunk)

            if i == len(sentence_boundaries) -1:
                temp_chunk = data[data["sentence_index"] > bound] 
                chunks.append(temp_chunk)


        num_successful_attacks_per_chunk = [len(c[c["attack_successful"] == True]) for c in chunks]
        num_attacks_to_return_per_chunk = [round(num_successful_attacks * percentage_attacks,0) for num_successful_attacks in num_successful_attacks_per_chunk]
        
        print(num_attacks_to_return_per_chunk)
        if sum(num_attacks_to_return_per_chunk) == 0:
            return None 

        sentence_indices = []
        for chunk_index, chunk in enumerate(chunks):
            best_attack_indices = []
            num_attacks_to_return = num_attacks_to_return_per_chunk[chunk_index]
            chunk_sentence_indices = []
            num_changed_list = sorted(chunk[chunk["attack_successful"] == True]["num_changed"].unique().tolist()) 

            for num_change in num_changed_list:
                best_attack_indices = chunk[(chunk["attack_successful"] == True) & (chunk["num_changed"] == num_change)].nlargest(int(num_attacks_to_return),keep="first", columns=["divergence"])

                for idx, attack in best_attack_indices.iterrows():
                    chunk_sentence_indices.append(attack["sentence_index"])
                
                if len(best_attack_indices) < num_attacks_to_return:
                    num_attacks_to_return = num_attacks_to_return - len(best_attack_indices)
                else:
                    break 
            
            if len(best_attack_indices) == 0:
                sentence_indices.append([None])
            else:
                sentence_indices.append(chunk_sentence_indices)
        
        return sentence_indices

    def __load_slot_indices__(self, disease_str):
        slot_index_path = os.path.join(dir_path, "Data/", "SlotIndices/")
        print(f"Loading Slot Indices from {slot_index_path}{disease_str}_SlotIndices.json")
        with open(os.path.join(slot_index_path, f"{disease_str}_SlotIndices.json"), "r") as slot_index_file:
            self.slot_indices = json.load(slot_index_file)

        with open(os.path.join(slot_index_path, f"{disease_str}_SlotIndicesReverse.json"), "r") as slot_index_file:
            self.slot_indices_reverse = json.load(slot_index_file)
            self.used_slots = list(self.slot_indices.keys())

                

def prepare_module(model_prefix, model, load_from_pickle, train, victim_type, comp_mode, load_weights, load_augmented=False, model_name=None, load_slot_indices=False):

    if model_prefix == "glaucoma":
        disease_str = "gl"
    else: 
        disease_str = "dm2"

    slot_filling_module = SlotFillingCompModule(tokenizer=tokenizer)
    
    load_data_in_prep = False

    #if load_from_pickle and not load_augmented:
    if (load_from_pickle and not load_augmented):
        print("importing documents from pickle")
        filepath_pickles = os.path.join(dir_path, "Data/", "Pickles/")
        documents, document_chunkings, batches = slot_filling_module.data_importer.import_from_pickle(disease_str=disease_str, dump_file_path=filepath_pickles)
        print(f"Normal: {len(documents[0])} for training")
        print(f"Normal: {len(documents[1])} for testing")
        print(f'number of slot: {batches[0][0]["entity_start_positions"].shape}')

        slot_filling_module.documents_train = documents[0]
        slot_filling_module.documents_test = documents[1]
        slot_filling_module.documents_validation = documents[2]

        slot_filling_module.document_chunkings_train = document_chunkings[0]
        slot_filling_module.document_chunkings_test = document_chunkings[1]
        slot_filling_module.document_chunkings_validation = document_chunkings[2]
        
    elif load_from_pickle and load_augmented:
        filepath_pickles = os.path.join(dir_path, "Data/", "Augmented/", "Only_Augmented_With_Unchanged/")
        # filepath_pickles = os.path.join(dir_path, "Data/", "Augmented/", "Only_Augmented/")
        # filepath_pickles = os.path.join(dir_path, "Data/", "Augmented/", "Appended/")
        
        documents, document_chunkings, batches = slot_filling_module.data_importer.import_augmented(disease_str=disease_str, model=model, 
                                                                                        victim_type=victim_type, comp_mode=comp_mode,
                                                                                        dump_file_path=filepath_pickles)
        if documents[0]:
            print(f"Augmented: {len(documents[0])} for training")
        else:
            print(f"Augmented: No train docs loaded")

        if documents[1]:
            print(f"Augmented: {len(documents[1])} for testing")
        else: 
            print(f"Augmented: No test docs loaded")
        slot_filling_module.documents_train = documents[0]
        slot_filling_module.documents_test = documents[1]

        slot_filling_module.document_chunkings_train = document_chunkings[0]
        slot_filling_module.document_chunkings_test = document_chunkings[1]
                
    else:
        print("Importing documents")
        documents = slot_filling_module.data_importer.import_documents(disease_str, only_train=False)
        slot_filling_module.documents_train = documents[0]
        slot_filling_module.documents_test = documents[1]
        slot_filling_module.documents_validation = documents[2]

        load_data_in_prep = True

    if load_slot_indices:
        slot_filling_module.__load_slot_indices__(disease_str=disease_str)
        slot_filling_module.batches_train = batches[0]
        slot_filling_module.batches_test = batches[1] 
        try:
            slot_filling_module.batches_validation = batches[2]
        except IndexError:
            pass 

    else:
        if not slot_filling_module.documents_train:
            filepath_pickles = os.path.join(dir_path, "Data/", "Pickles/")
            documents, document_chunkings, _ = slot_filling_module.data_importer.import_from_pickle(disease_str=disease_str, dump_file_path=filepath_pickles)
            slot_filling_module.documents_train = documents[0]
            slot_filling_module.document_chunkings_train = document_chunkings[0]
        
        slot_filling_module.create_slot_indices(MIN_SLOT_FREQ)
        document_encoder = DocumentEncoder(tokenizer, slot_filling_module.slot_indices, 'no_slot')
        if slot_filling_module.documents_train:
            slot_filling_module.batches_train = create_batches(slot_filling_module.documents_train, slot_filling_module.document_chunkings_train, document_encoder, slot_filling_module.slot_indices)
        if slot_filling_module.document_chunkings_test:
            slot_filling_module.batches_test = create_batches(slot_filling_module.documents_test, slot_filling_module.document_chunkings_test, document_encoder, slot_filling_module.slot_indices)
            
    load_optimizer = False         
    if train:
        load_optimizer = True 

    slot_filling_path = os.path.join(dir_path, "Data/", "SlotIndices/")
    slot_filling_module.prepare(optimizer=load_optimizer, disease_str=disease_str, load_data=load_data_in_prep, slot_index_path=slot_filling_path)
    
    if load_weights:
        if not model_name:
            filepath_model = os.path.join(dir_path, "slot_filling_models/", f"{disease_str}/", f"{model_prefix}_{EPOCHS}epochs")
        else:
            filepath_model = os.path.join(dir_path, "slot_filling_models/", f"{disease_str}/", model_name)
        
        print(f"loading model at {filepath_model}")
        slot_filling_module.load_model_weights(filepath_model)
        
    return slot_filling_module 

def import_attacked(slot_filling_module, disease_str, model, victim_type, comp_mode, as_attacked=True):
    attacked_path = "/home/lena/TemplateFilling/Data/Attacked"
    document_chunkings_train_attacked, document_chunkings_test_attacked, batches_train_attacked, batches_test_attacked = slot_filling_module.data_importer.import_attacked(model=model, 
                                                            disease_str=disease_str, victim_type=victim_type, comp_mode=comp_mode, dump_file_path=attacked_path)
    

    if as_attacked:
        slot_filling_module.document_chunkings_train_attacked = document_chunkings_train_attacked
        slot_filling_module.document_chunkings_test_attacked = document_chunkings_test_attacked
        slot_filling_module.batches_train_attacked = batches_train_attacked
        slot_filling_module.batches_test_attacked = batches_test_attacked
    else:
        slot_filling_module.document_chunkings_train = document_chunkings_train_attacked
        slot_filling_module.document_chunkings_test = document_chunkings_test_attacked
        #slot_filling_module.batches_train = batches_train_attacked
        #slot_filling_module.batches_test = batches_test_attacked

        document_encoder = DocumentEncoder(tokenizer, slot_filling_module.slot_indices, 'no_slot')
        slot_filling_module.batches_train = create_batches(slot_filling_module.documents_train, slot_filling_module.document_chunkings_train, document_encoder, slot_filling_module.slot_indices)
        slot_filling_module.batches_test = create_batches(slot_filling_module.documents_test, slot_filling_module.document_chunkings_test, document_encoder, slot_filling_module.slot_indices)
        

if __name__ == '__main__':
    disease_str, victim_type, start, end, comp_mode, data_mode, model = parse_cli_arguments(for_slot_filling=True)
    if model.lower() == "clare":
        model = model.upper() 
    else:
        model=model.capitalize() 

    if disease_str == "gl":
        model_prefix = "glaucoma"
    else:
        model_prefix = disease_str

    load_augmented = False
    load_slot_indices = True 
    model_name = None

    slot_filling_module = prepare_module(model_prefix=model_prefix, load_from_pickle=True, train=True, load_weights=False, model=model, 
                                            comp_mode=comp_mode, load_augmented=load_augmented, victim_type=victim_type, 
                                            load_slot_indices=load_slot_indices, model_name=model_name)
    


    exit()
    model_descr = f"{model}_{victim_type}_augmented" if victim_type == "extraction" else f"{model}_{victim_type}_{comp_mode}_augmented"
    slot_filling_module.train(num_epochs=EPOCHS, model_descr=model_descr, model_prefix=model_prefix)

    exit()
    
    with open(f"/home/lena/TemplateFilling/Data/SlotIndices/DocumentLength_train_{disease_str}.csv", "w") as outfile:
        writer = csv.writer(outfile)
        data = ["abstract_id", "num_sentences", "num_tokens", "num_entities"]
        writer.writerow(data)
        data = []
        for doc in slot_filling_module.documents_train:
            data.append(doc._abstract.abstract_id)
            data.append(doc.get_num_sentences())
            num_tokens = sum([s.get_num_tokens() for s in doc.get_sentences()])
            data.append(num_tokens)
            num_entities = sum([s.get_num_entities() for s in doc.get_sentences()])
            data.append(num_entities)
            writer.writerow(data)
            data = []

    with open(f"/home/lena/TemplateFilling/Data/SlotIndices/DocumentLength_test_{disease_str}.csv", "w") as outfile:
        writer = csv.writer(outfile)
        data = ["abstract_id", "num_sentences", "num_tokens", "num_entities"]
        writer.writerow(data)
        data = []
        for doc in slot_filling_module.documents_test:
            data.append(doc._abstract.abstract_id)
            data.append(doc.get_num_sentences())
            num_tokens = sum([s.get_num_tokens() for s in doc.get_sentences()])
            data.append(num_tokens)
            num_entities = sum([s.get_num_entities() for s in doc.get_sentences()])
            data.append(num_entities)
            writer.writerow(data)
            data = []