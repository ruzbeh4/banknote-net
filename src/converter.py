import os
# Force TensorFlow to use the older Keras 2 engine to read the old .h5 file
os.environ['TF_USE_LEGACY_KERAS'] = '1'


import tensorflow as tf

# 1. Load the actual Microsoft Keras model from their repo folder
# (Make sure this path points to the exact .h5 file you want to use)
address = './custom_classifier'
keras_model_path = f'{address}.h5'
model = tf.keras.models.load_model(keras_model_path)

# 2. Tell TensorFlow to convert it
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Apply the Quantization compression to make it small for Flutter
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. Do the conversion!
tflite_model = converter.convert()

# 5. Save the final file that you will drag into your Flutter app
with open(f'{address}.tflite', 'wb') as f:
    f.write(tflite_model)

print("Boom! Conversion complete.")