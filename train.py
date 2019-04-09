import numpy as np
from keras import models
from keras import layers

def vectorize_sequences(sequences,dimension=3):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

def to_one_hot(labels,dimension=27):
    results = np.zeros((len(labels),dimension))
    for i,label in enumerate(labels):
        results[i, label] = 1
    return results

data = vectorize_sequences(np.loadtxt("data.csv", delimiter=",",dtype="int"),3)
result = to_one_hot(np.loadtxt("result.csv",delimiter=",",dtype="int"),27)

model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(3,)))
model.add(layers.Dense(27,activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(data,result,epochs=9,batch_size=512)
predictions = model.predict(vectorize_sequences([[2,39600,5]],3))
print predictions



