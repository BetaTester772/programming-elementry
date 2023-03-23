# tensorflow 불러오기
import tensorflow as tf

#mnist만들기
mnist = tf.keras.datasets.mnist

# make x_train, y_train, x_test, y_test
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x trina / 255
x_train, x_test = x_train / 255.0, x_test / 255.0

# 모델 만들기
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 모델 컴파일 만들기
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# 모델 fit 만들기
history = model.fit(x_train, y_train, epochs=5)

# 모델 evaluate 만들기
model.evaluate(x_test, y_test)

# 정확도 그래프 만들기
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
