import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

# 1. Подготовка данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Построение модели
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 3. Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Обучение модели
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# 5. Оценка модели
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Точность на тестовых данных: {test_accuracy:.2f}")

# 6. Прогнозирование
sample_image = x_test[0].reshape(1, 28, 28)
predicted_class = np.argmax(model.predict(sample_image))
print(f"Предсказанный класс: {predicted_class}")
