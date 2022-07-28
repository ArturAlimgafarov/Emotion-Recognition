# подключение библиотек
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory

# подключение гугл диска с обучающими и тестовыми наборами
from google.colab import drive, files
drive.mount('Datasets')

# параметры наборов
seed_value = 1206
split_value = 0.2
btc_size = 256
img_size = (48, 48)
datasets_path = 'Datasets/MyDrive/Colab Notebooks/archive'

# тренировочный набор
training_dataset = image_dataset_from_directory(datasets_path + '/train',
                                                subset='training',
                                                seed=seed_value,
                                                validation_split=split_value,
                                                batch_size=btc_size,
                                                image_size=img_size)

# проверочный набор
validation_dataset = image_dataset_from_directory(datasets_path + '/train',
                                                subset='validation',
                                                seed=seed_value,
                                                validation_split=split_value,
                                                batch_size=btc_size,
                                                image_size=img_size)

# тестовый набор
test_dataset = image_dataset_from_directory(datasets_path + '/test',
                                            batch_size=btc_size,
                                            image_size=img_size)

# классы распознавания (выходные значения сети)
class_names = training_dataset.class_names
print(class_names)

# настройка производительности для увеличения скорости обечения, вызов prefetch
# для предварительной загрузки
AUTOTUNE = tf.data.experimental.AUTOTUNE

training_dataset = training_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# Создаем последовательную модель
model = Sequential()

# Сверточный слой
model.add(Conv2D(16, (5, 5), activation='relu',
                 input_shape=(img_size[0], img_size[1], 3), padding='same'))

# Слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))

# Сверточный слой
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))

# Слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))

# Сверточный слой
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))

# Слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))

# Сверточный слой
model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))

# Слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))

# Полносвязная часть нейронной сети для классификации
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

# Выходной слой, 4 нейрон по количеству классов
out_count = len(class_names)
model.add(Dense(out_count, activation='softmax'))

model.summary()

# компиляция модели
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# запуск процесса обучения
epochs_count = 5
history = model.fit(training_dataset,
                    validation_data=validation_dataset,
                    epochs=epochs_count,
                    verbose=1)

# оценка качества работы нейросети на тестовом наборе
scores = model.evaluate(test_dataset, verbose=1)

# сохранение модели и загрузка на ПК
nn_name = input('Enter the neural network model name: ') + '.h5'

model.save(nn_name)
files.download(nn_name)