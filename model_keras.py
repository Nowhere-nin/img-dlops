import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# Definir las categorías
categories = ['auto', 'casa', 'edificio', 'gato', 'moto']

# Definir las dimensiones de las imagenes
img_height = 224
img_width = 224

# Creación de un diccionario para mapear los nombres de las imagenes a las categorias
category_map = {}
for img_name in os.listdir('dataset/train'):
    for category in categories:
        if category in img_name:
            category_map[img_name] = categories.index(category)
            break


# Creación de un diccionario para mapear los nombres de las imagenes a las categorias
category_map2 = {}
for img_name in os.listdir('dataset/test'):
    for category in categories:
        if category in img_name:
            category_map2[img_name] = categories.index(category)
            break


# Carga de las imagenes de entrenamiento
X_train = []
y_train = []
for img_name in os.listdir('dataset/train'):
    img = load_img(os.path.join('dataset/train', img_name), target_size=(img_height, img_width))
    img_array = img_to_array(img)
    X_train.append(img_array)
    y_train.append(category_map[img_name])

X_train = np.array(X_train)
y_train = np.array(y_train)

y_train = to_categorical(y_train, num_classes=len(categories))

# Carga de las imagenes de test
X_test = []
y_test = []
for img_name in os.listdir('dataset/test'):
    img = load_img(os.path.join('dataset/test', img_name), target_size=(img_height, img_width))
    img_array = img_to_array(img)
    X_test.append(img_array)
    y_test.append(category_map2[img_name])

X_test = np.array(X_test)
y_test = np.array(y_test)

y_test = to_categorical(y_test, num_classes=len(categories))

# Definir la forma de entrada
input_shape = (img_height, img_width, 3)

# Definir la capa de entrada
input_layer = Input(shape=input_shape)

# Definir las capas convolucionales
x = Conv2D(512, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

# Definir la capa Flatten
x = Flatten()(x)

# Definición de las capas densas
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(len(categories), activation='softmax')(x)


# Definir la capa de salida
output_layer = x

# Creación del modelo
model = Model(inputs=input_layer, outputs=output_layer)

# Compilación del modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(X_train, y_train, epochs=160, batch_size=32, validation_split=0.2)

# Evaluar el modelo con datos de test para obtener información aproximada en cuanto al funcionamiento del modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.2f}')
print(f'Test loss: {loss:.2f}')

# Guardar el modelo entrenado en un archivo .h5
model.save('models/mi_modelo.h5')