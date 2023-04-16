# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 11:51:10 2023

@author: Mehmet ÖZTÜRK
"""

import cv2  # OpenCV kütüphanesi
import pickle  # Python'da nesneleri seri hale getirip kaydedebilmek için kullanılan kütüphane
import tkinter as tk  # Grafik kullanıcı arayüzü oluşturma kütüphanesi

import numpy as np  # Sayısal hesaplama kütüphanesi
from PIL import Image, ImageTk  # Görüntü işleme için kullanılan kütüphane

# Eğitilmiş modeli yükle
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Tkinter penceresi oluştur
root = tk.Tk()
root.title("Yangın Tespit Sistemi")

# Video oynatıcısı için canvas oluştur
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

# Yangın tespit edildiğinde görüntülenecek etiket
label = tk.Label(root, text="Güvenli ortam. İyi günler dileriz.", fg="green")
label.place(x=240, y=450)

# Varsayılan kaydırma boyutları
window_size = (100, 100)
step_size = 50


# Varsayılan özellik çıkarma fonksiyonu
def extract_features(image):
    # Örnek olarak RGB renklerin ortalamasını alıyoruz
    return [image[:, :, i].mean() for i in range(3)]


# Varsayılan kaydırma fonksiyonu
def sliding_window(frame, window_size=(100, 100), step_size=50):
    coordinates = []
    for y in range(0, frame.shape[0] - window_size[1], step_size):
        for x in range(0, frame.shape[1] - window_size[0], step_size):
            coordinates.append((x, y, x + window_size[0], y + window_size[1]))
    return coordinates

from tensorflow.keras.preprocessing import image
# Giriş görüntüsü dosyaları için sabitler
img_folder_path = "fotos/"
img_extension = ".jpg"

# Giriş görüntüsü dosyaları için rastgele sıralama yapılır
import os

file_names = sorted(os.listdir(img_folder_path))
import random

random.shuffle(file_names)

# Video üzerindeki her bir kare için    
while True:
    for img in file_names:
        string1 = img_folder_path + img
        frame = cv2.imread(string1)
    
        img = image.load_img(string1, target_size=(224, 224))
        x = image.img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
    
        # Yangın tespiti yapmak için, resim üzerinde bir kaydırma penceresi oluşturulur
        coordinates = sliding_window(frame, window_size, step_size)
    
        # Başlangıçta yangın tespit edilmediği varsayılır
        fire_detected = False
    
        # Her pencere üzerinde dolaşarak yangın tespiti yap
        for (startX, startY, endX, endY) in coordinates:
    
            # Pencere boyutunda bir bölge resmi kırp
            cropped_img = frame[startY:endY, startX:endX]
    
            # Resimden özellikler (renk, yoğunluk, vs.) çıkar
            features = extract_features(cropped_img)
    
            # Kırpılmış resmi bir ölçekte yeniden boyutlandır
            resized_frame = cv2.resize(frame, (224, 224))
    
            # Yeniden boyutlandırılmış resmi model için hazırla
            input_image = np.expand_dims(resized_frame, axis=0) / 255.0
    
            # Model üzerinde tahmin yap
            prediction = model.predict(x)
    
            # Eğer tahmin olasılığı 0.5'in altındaysa, pencereyi kırmızı çerçeve içine al ve yangın tespiti yapıldı olarak işaretle
            if prediction < 0.5:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                fire_detected = True
    
        # Çerçeveyi renk formatından PIL formatına dönüştür
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((640, 450), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
    
        # Canvas'a resmi yerleştir
        canvas.create_image(0, 0, image=img, anchor=tk.NW)
        canvas.img = img
    
        # Eğer yangın tespit edildiyse etiketi güncelle
        if fire_detected:
            label.config(text="Dikkat! Yangın Çıktı! İtfayeye Haber Veriliyor.", fg="red")
        else:
            label.config(text="Güvenli ortam. İyi günler dileriz.", fg="green")
    
        # Pencereyi güncelle
        root.update()
    
cv2.waitKey(0)
cv2.destroyAllWindows()
