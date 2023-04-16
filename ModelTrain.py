import tensorflow as tf
import pickle
import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression

# Yapay sinir ağı modeli oluşturma
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Modeli derleme
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Veri artırma işlemleri
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Eğitim ve test veri setlerini yükleyin
train_set = train_datagen.flow_from_directory('dataset/train', target_size=(224, 224), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory('dataset/test', target_size=(224, 224), batch_size=32, class_mode='binary')

# Yapay sinir ağını eğitme
model.fit(train_set, validation_data=test_set, epochs=3)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
    

"""

tf.keras.utils.plot_model(
    model,
    to_file="model2.png",
    show_shapes=False,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=True,
    dpi=96,
    layer_range=None,
    show_layer_activations=True,
    show_trainable=False,
)
"""