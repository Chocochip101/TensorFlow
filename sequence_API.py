from tensorflow import keras

#패션 mnist 데이터셋 적재
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

#검증 세트 설정 & 특성 스케일(0~1로 정규화)
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

#레이블에 해당하는 아이템 이름
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress','Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

'''
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
'''

#summary() 메서드를 통한 모델의 모든 층 출력
print(model.summary())

#layers 메서드를 통한 모델의 모든 층 출력
for i in model.layers:
    print(i)
layer_1st = model.layers[1]
print(layer_1st.name)

#첫 Dense층의 가중치와 편향 출력
layer_1st = model.layers[1]
weights, bias = layer_1st.get_weights()
print(weights)
print(weights.shape)
print(bias)
print(bias.shape)
