from flask import Flask, request, jsonify
import onnxruntime as rt
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Cargar el modelo ONNX
sess = rt.InferenceSession("models/convert/mi_modelo.onnx")

# Se define el tamaño de la imagen de entrada
img_height = 224
img_width = 224

# Se definen las categorías
categories = ['auto', 'casa', 'edificio', 'gato', 'moto']

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    # Convertir la imagen a un array numpy
    img = Image.open(io.BytesIO(file.read())).resize((img_width, img_height))
    img_array = np.array(img).astype('float32')
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)

    # Se realiza la predicción
    input_name = sess.get_inputs()[0].name
    result = sess.run(None, {input_name: img_array})

    # Se obtiene la categoría con la probabilidad más alta
    prediction = np.argmax(result[0], axis=1)[0]
    print(prediction)
    predicted_category = categories[prediction]

    return jsonify({'prediction': predicted_category})

if __name__ == '__main__':
    app.run(debug=True)