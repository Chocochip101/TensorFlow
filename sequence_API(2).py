import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow import keras

# 패션 mnist 데이터셋 적재
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# 검증 세트 설정 & 특성 스케일(0~1로 정규화)
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

# 레이블에 해당하는 아이템 이름
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# 모델 컴파일
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

# pandas를 활용한 학습 곡선 출력
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# 모델 평가
print(model.evaluate(X_test, y_test))

# 모델 예측
X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))

print()

y_pred = model.predict_classes(X_new)
print(y_pred)
print(np.array(class_names)[y_pred])
print()

#정답 레이블
y_new = y_test[:3]
print(y_new)
