import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

# Modeli yükle
from tensorflow.keras.models import load_model
model = load_model('my_model_V2.h5')






img_folder_path = 'C:\\Users\\Mahmut\\Desktop\\test2\\'
test_images = []
test_labels = []

# Giriş görüntüsü dosyaları için rastgele sıralama yapılır
import os
import keras.utils as image

file_names = sorted(os.listdir(img_folder_path + "fire\\"))

for img in file_names:
    image_path = os.path.join(img_folder_path + "fire\\", img)

    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    test_images.append(x)
    test_labels.append(0) #fire



file_names = sorted(os.listdir(img_folder_path + "nofire\\"))

for img in file_names:
    image_path = os.path.join(img_folder_path + "nofire\\", img)

    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    test_images.append(x)
    test_labels.append(1) #fire


y_pred = []
sonuc = []
for image2 in test_images:
    prediction = model.predict(image2)
    sonuc.append(prediction)
    if prediction < 0.5:
        y_pred.append(0)
    else:
        y_pred.append(1)

print(sonuc)
#y_pred2 = np.argmax(y_pred, axis=1)

# Karışıklık matrisini hesapla
cm = confusion_matrix(test_labels, y_pred)
print(cm)




import seaborn as sns
import matplotlib.pyplot as plt
print('Confusion Matrix:\n', cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()



from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(test_labels, y_pred)
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






"""
# Sınıf etiketlerini belirle
classes = ['fire', 'nofire']

# Test verilerinin dizini (klasörü)


# Test verilerini ve etiketlerini saklamak için boş listeler oluştur
test_images = []
test_labels = []


import keras.utils as image

# sinif_a klasöründeki fotoğrafları yükle ve etiketlerini ayarla
fire_path = os.path.join(test_data_dir, 'fire')
for image_name in os.listdir(fire_path):
    image_path = os.path.join(fire_path, image_name)


    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    #image = cv2.imread(image_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV varsayılan olarak BGR formatında açar, RGB'ye dönüştürüyoruz
    #image = cv2.resize(image, (224, 224))  # Gerekirse boyutunu ayarlayın
    test_images.append(x)
    test_labels.append(0)  # sinif_a için etiket 0

# sinif_b klasöründeki fotoğrafları yükle ve etiketlerini ayarla
nofire_path = os.path.join(test_data_dir, 'nofire')
for image_name in os.listdir(nofire_path):
    image_path = os.path.join(nofire_path, image_name)

    img = image.load_img(image_path, target_size=(224, 224))


    #image = cv2.imread(image_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = cv2.resize(image, (224, 224))
    test_images.append(x)
    test_labels.append(1)  # sinif_b için etiket 1

# Test verilerini numpy dizilerine dönüştür
test_images = np.array(test_images)
test_labels = np.array(test_labels)




# Test verilerini normalize et (örnek olarak)
#test_images = test_images / 255.0

# Tahminleri yap
#y_pred_probs = model.predict(test_images)
#y_pred = np.argmax(y_pred_probs, axis=1)

y_pred = []
sonuc = []
for image2 in test_images:


    prediction = model.predict(image2)
    sonuc.append(prediction)
    if prediction < 0.5:
        y_pred.append(0)
    else:
        y_pred.append(1)

print(sonuc)
#y_pred2 = np.argmax(y_pred, axis=1)

# Karışıklık matrisini hesapla
cm = confusion_matrix(test_labels, y_pred)

# Karışıklık matrisini görselleştir
# (yukarıdaki kod örneğindeki gibi)

"""








"""


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
loaded_model = keras.models.load_model('my_model_V2.h5')

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


"""