import numpy as np
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical


def vectorize_sequences(sequences,dimension=100000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

data = vectorize_sequences(np.loadtxt("data.csv", delimiter=",",dtype="int"),100000)
result = to_categorical(np.loadtxt("result.csv",delimiter=",",dtype="int"))

model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(100000,)))
#model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(27,activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(data,result,epochs=9,batch_size=512)
predictions = model.predict(vectorize_sequences([[0,39600,110]],100000))
print predictions



