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



train_loss = history['loss']
val_loss = history['val_loss']

# grafiği çizdirme
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'g', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
