import tensorflow as tf
model_path = "final_model_structure.pb"
model = tf.saved_model.load(model_path)