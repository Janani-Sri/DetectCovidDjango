import tensorflow as tf
from keras.models import load_model

graph = tf.get_default_graph()
model = load_model('classify/model.hdf5')

output_dict = {'covid': 0,
               'normal': 1,
               'viral': 2}

output_list = list(output_dict.keys())