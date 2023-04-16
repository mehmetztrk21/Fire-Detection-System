import cv2
import pickle
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

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
label = tk.Label(root, text="")
label.pack()

# Videoyu aç
video = cv2.VideoCapture('video.mp4')
video.read()
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

# Her kareyi oku ve üzerinde yangın olup olmadığını kontrol et
while True:
    ret, frame = video.read()
    for tekrar in range(36):
        ret, frame = video.read()
    if not ret:
        break
    coordinates = sliding_window(frame, window_size, step_size)
    fire_detected = False
    for (startX, startY, endX, endY) in coordinates:
        cropped_img = frame[startY:endY, startX:endX]
        features = extract_features(cropped_img)

        resized_frame = cv2.resize(frame, (224, 224))
        input_image = np.expand_dims(resized_frame, axis=0) / 255.0
        prediction = model.predict(input_image)
        if prediction == 1:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            fire_detected = True
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, image=img, anchor=tk.NW)
    canvas.img = img
    if fire_detected:
        label.config(text="Dikkat! Yangın Çıktı! İtfayeye Haber Veriliyor", fg="red")
    else:
        label.config(text="")
    root.update()

# Kaynakları serbest bırak
video.release()
cv2.destroyAllWindows()
