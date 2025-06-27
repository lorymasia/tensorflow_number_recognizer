import tensorflow as tf                   # Importa TensorFlow
import tensorflow_datasets as tfds        # Importa TensorFlow Datasets
import matplotlib.pyplot as plt           # Importa Matplotlib per i grafici
from tensorflow.keras.utils import plot_model   # Funzione per visualizzare il modello # type: ignore
from tensorflow.keras.callbacks import TensorBoard  # Callback per TensorBoard # type: ignore

print("tensorflow version = ", tf.__version__)      # Stampa la versione di TensorFlow
print("tensorflow_datasets version = ", tfds.__version__)  # Stampa la versione di tfds
gpus = tf.config.list_physical_devices('GPU')       # Controlla se ci sono GPU disponibili
if gpus:
    print("GPU trovata:")
    for gpu in gpus:
        print(" -", gpu)
else:
    print("Nessuna GPU trovata, si userà la CPU.")

# Carica il dataset MNIST (immagini di cifre scritte a mano)
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files = True,
    as_supervised = True,
    with_info = True
)

def normalize_image(img, label):
    return tf.cast(img, tf.float32) / 255., label   # Normalizza le immagini tra 0 e 1

# Prepara il dataset di training: normalizza, memorizza in cache, mescola, crea batch e prefetch
ds_train = ds_train.map(normalize_image, num_parallel_calls = tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

# Prepara il dataset di test: normalizza, crea batch, memorizza in cache e prefetch
ds_test = ds_test.map(normalize_image, num_parallel_calls = tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

tensorboard_callback = TensorBoard(log_dir = './classificatore_cifre/logs')  # Callback per TensorBoard

# Definisce il modello sequenziale: Flatten, Dense, Dropout, Dense finale
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),      # Appiattisce l'immagine 28x28 in un vettore
    tf.keras.layers.Dense(512, activation = 'relu'),      # Primo strato denso con ReLU
    tf.keras.layers.Dropout(0.2),                        # Dropout per ridurre overfitting
    tf.keras.layers.Dense(10)                            # Strato di output (10 classi)
])

model.summary()   # Mostra un riepilogo del modello

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)  # Funzione di perdita

# Compila il modello con ottimizzatore Adam, funzione di perdita e accuratezza come metrica
model.compile(
    optimizer = 'adam',
    loss = loss_fn,
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
)

# Allena il modello per 40 epoche, usando il dataset di training e validazione, con TensorBoard
history = model.fit(
    ds_train,
    epochs = 40,
    validation_data = ds_test,
    shuffle = False,
    callbacks = [tensorboard_callback]
)

model.evaluate(ds_test, verbose = 2)   # Valuta il modello sul dataset di test
model.save("./classificatore_cifre/mio_modello.keras")   # Salva il modello addestrato

# Estrae le metriche di loss e accuracy dal training e dalla validazione
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

training_accuracy = history.history['sparse_categorical_accuracy']
validation_accuracy = history.history['val_sparse_categorical_accuracy']

# Grafico della loss (errore) durante il training e la validazione
plt.subplot(1,2,1)
plt.plot(training_loss, label="Training loss")
plt.plot(validation_loss, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and validation loss")
plt.legend()
plt.grid(True)

# Grafico dell'accuracy (precisione) durante il training e la validazione
plt.subplot(1,2,2)
plt.plot(training_accuracy, label="Training accuracy")
plt.plot(validation_accuracy, label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()   # Mostra

