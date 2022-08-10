import sys
import tensorflow as tf
import tensorflow_hub as hub
from official.nlp.bert import tokenization
from official.nlp import optimization

from ctro import *
from Document import *


bert_model_name = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/2"


class EntityTagger(tf.keras.Model):
    def __init__(self):
        super(EntityTagger, self).__init__(self)
        
        #self.bert_layer = hub.KerasLayer(bert_model_name, trainable=True)
        self.l = tf.keras.layers.Dense(10)
        
        
entity_tagger = EntityTagger()
entity_tagger.load_weights('model')