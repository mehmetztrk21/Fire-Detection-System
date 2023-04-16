from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Veri artırma işlemleri
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Eğitim ve test veri setlerini yükleyin
test_set = test_datagen.flow_from_directory('dataset/test', target_size=(224, 224), batch_size=32, class_mode='binary')



from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import keras

# Modeli yükle
loaded_model = keras.models.load_model('my_model.h5')

# Test verilerini yükle
test_data = test_set
y_true = test_set.labels

# Tahminleri yap
y_pred = loaded_model.predict(test_data)

# ROC eğrisi oluştur
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
