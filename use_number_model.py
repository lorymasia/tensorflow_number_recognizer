import tensorflow as tf                   # Importa TensorFlow
import numpy as np                        # Importa NumPy per operazioni numeriche
import matplotlib.pyplot as plt           # Importa Matplotlib per visualizzare immagini
from tensorflow.keras.preprocessing import image # Funzioni utili per immagini (non usate qui) # type: ignore
import sys
from PIL import Image as ImagePl          # Importa PIL per immagini (non usato qui)

model_path = "classificatore_cifre/mio_modello.keras"    # Percorso del modello salvato
model = tf.keras.models.load_model(model_path)           # Carica il modello addestrato

class_names = ["0","1","2","3","4","5","6","7","8","9"]  # Nomi delle classi (cifre)

def preprocess_image(img_path, img_height=28, img_wide=28):
    img = tf.io.read_file(img_path)                      # Legge il file immagine
    img = tf.image.decode_jpeg(img, channels=3)          # Decodifica JPEG in tensore
    img = tf.image.resize(img, (28, 28))                 # Ridimensiona a 28x28 pixel
    gray = tf.image.rgb_to_grayscale(img)                # Converte in scala di grigi

    plt.imshow(gray, cmap="gray")                        # Mostra l'immagine
    plt.axis("off")
    plt.show()

    print(f"shape= {gray.shape}")                        # Stampa la forma del tensore
    gray = tf.squeeze(gray, axis= -1)                    # Rimuove la dimensione canale
    print(f"shape= {gray.shape}")
    gray = tf.expand_dims(gray, axis=0)                  # Aggiunge dimensione batch
    print(f"shape= {gray.shape}")

    return gray                                          # Restituisce il tensore pronto

def make_prediction(image_path):
    img_array = preprocess_image(image_path)             # Pre-elabora l'immagine
    predictions = model.predict(img_array)               # Ottiene le predizioni dal modello
    predicted_class_idx = np.argmax(predictions, axis=1)[0]  # Trova la classe con probabilità massima
    confidence = predictions[0][predicted_class_idx]         # Ottiene la confidenza
    predicted_class = class_names[predicted_class_idx]       # Ottiene il nome della classe

    return predicted_class, confidence                   # Restituisce classe e confidenza

for i in range(10):
    img_path = f"classificatore_cifre/test/mio_numero_{i}.jpg"   # Percorso dell'immagine di test
    predicted_class, confidence = make_prediction(img_path)      # Predice la classe
    print(f"Classe predetta: {predicted_class} con confidenza: {confidence / 100:.2f}%")  # Stampa il risultato
