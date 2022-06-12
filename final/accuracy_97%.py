####정확도 97프로가 나왔다.....
import pickle
import pandas as pd
import numpy as np
from keras.utils import to_categorical, np_utils
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Dropout





# 교수님이 주신 테스트셋 적용
with open('./test_data/testdata1D.pkl', 'rb') as f:
    datasetL = pickle.load(f)
x_test, y_test = datasetL


# train 데이터셋을 csv로 변환한 것
train=pd.read_csv('./mnist_train.csv')


# 행단위로 적용된다.
x_train = train.drop('label', axis=1)
y_train = train['label'] #target


# 입력을 0~255에서 0~1로 정규화
x_train=x_train/255

# CNN 모델 입력층에 맞춰서 reshape하는 과정
x_train = np.array(x_train).reshape((-1, 28, 28, 1))
x_test = x_test.reshape(429, 28, 28, 1).astype('float32') / 255.0

# one-hot 인코딩 처리를 수행 , num_classes은 최종 출력 클래스의 크기
y_test = np_utils.to_categorical(y_test)
y_train = to_categorical(y_train, num_classes=10)

# 형태 확인
print(x_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train.shape)

# 모델 구성


model = keras.Sequential([
    keras.layers.Conv2D(kernel_size=(3,3), filters=75, input_shape=(28,28,1), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Conv2D(kernel_size=(3,3), filters=50, padding='same', activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.MaxPool2D(pool_size=(2,2)),

    keras.layers.Conv2D(kernel_size=(3,3), filters=25, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),

    keras.layers.Flatten(),
    keras.layers.Dense(units=512, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(units=10, activation='softmax')
])


# loss는 현재 가중치 세트를 평가하는데 사용한 손실 함수 , optimizer은 최적화 알고리즘, metrics는 평가 척도를 나타냄, 학습률은 0.0002
#model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002), loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

# 모델 학습시키기
history = model.fit(x_train, y_train, batch_size= 256 , epochs=15, validation_data=(x_test, y_test))

# 모델 저장
model.save('deep3.h5')

# 행렬의 축을 지정, 분류한 결과를 확인
pred = np.argmax(model.predict(x_train), axis=-1)


# 모델 평가하기
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print("손실률:", loss)
print("정확도:", acc)


# 에포크별 학습 지표를 그래프로 나타냄
plt.figure(figsize=(18, 6))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.title("accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("loss")
plt.legend()

plt.show()

