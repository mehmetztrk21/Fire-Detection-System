from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Veri artırma işlemleri
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Eğitim ve test veri setlerini yükleyin
test_set = test_datagen.flow_from_directory('dataset/test', target_size=(224, 224), batch_size=32, class_mode='binary')


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import keras

# Modeli yükle
loaded_model = keras.models.load_model('my_model.h5')

# Test verilerini yükle
test_data = test_set
y_true = test_set.labels

# Tahminleri yap
y_pred = loaded_model.predict(test_data)

# Confusion matrix oluştur
cm = confusion_matrix(y_true, y_pred.round())
print('Confusion Matrix:\n', cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
