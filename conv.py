import tensorflow as tf
import tf2onnx

# Define the image dimensions
img_height = 224
img_width = 224

# Carga el modelo desde el archivo .h5
model = tf.keras.models.load_model('mi_modelo.h5')

# Define la especificación de entrada
spec = (tf.TensorSpec((None, img_height, img_width, 3), tf.float32, name="input"),)

# Define el nombre del archivo de salida
output_path = "mi_modelo1.onnx"

# Convierte el modelo de Keras a formato ONNX
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Guarda el modelo ONNX en el archivo especificado
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f'Modelo guardado en {output_path}')