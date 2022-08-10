import tensorflow as tf
import tensorflow_hub as hub
#from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Embedding,Conv1D,Dense,Dropout



class SlotFillingModel(tf.keras.Model):
    def __init__(self, bert_model_name, bert_model_dim, num_entity_labels, dropout_rate=0.1):
        super(SlotFillingModel, self).__init__()
        self.bert_model_dim = bert_model_dim
        
        # BERT encoder
        self.bert_layer = hub.KerasLayer(bert_model_name, trainable=True)
        
        # entity representation layer
        self.dense_entity_representation = Dense(bert_model_dim, activation='relu')
        self.dropout_entity_representation = Dropout(dropout_rate)
        
        # dense layers for predicting start/end positions of entities
        self.dense_entity_start_positions = Dense(num_entity_labels)
        self.dense_entity_end_positions = Dense(num_entity_labels)
        
        # dense layer for entity compatibility prediction
        self.dense_entity_compatibility = Dense(1)
        
        
    def compute_entity_representation(self, encoder_output, entity_start_token_indices, entity_end_token_indices, training=False):
        start_token_vectors = tf.gather_nd(encoder_output, entity_start_token_indices)
        end_token_vectors = tf.gather_nd(encoder_output, entity_end_token_indices)
        #print(tf.shape(start_token_vectors))
        
        
        entity_representations = (start_token_vectors + end_token_vectors) / 2.0
        entity_representations = tf.reshape(entity_representations, shape=(-1, self.bert_model_dim))
        entity_representations = self.dense_entity_representation(entity_representations)
        entity_representations = self.dropout_entity_representation(entity_representations, training=training)
        
        return entity_representations

    
if __name__ == "__main__":
    BERT_MODEL_DIM = 768
    slot_filling_model = SlotFillingModel(bert_model_name="https://tfhub.dev/google/experts/bert/pubmed/2",bert_model_dim=BERT_MODEL_DIM,num_entity_labels=37)
    print(slot_filling_model.layers)