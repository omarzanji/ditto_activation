import tensorflow as tf

print('\n[Converting Keras model to TFLite...]\n')

# model = tf.keras.models.load_model('models/HeyDittoNet_CNN-LSTM')
model = tf.keras.models.load_model('models/HeyDittoNet-v1')

converter = tf.lite.TFLiteConverter.from_keras_model(model)


converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

tfmodel = converter.convert()
open('models/model.tflite', 'wb').write(tfmodel)

print('\n[Done! Saved to "models/model.tflite"]\n')
