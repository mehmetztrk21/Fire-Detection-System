import matplotlib.pyplot as plt
import numpy as np  # Sayısal hesaplama kütüphanesi

history = np.load('my_history.npy', allow_pickle=True).item()

print(history.keys())
accuracy = history['acc']
val_accuracy = history['val_acc']

plt.plot(accuracy)
plt.plot(val_accuracy)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()