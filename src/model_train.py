from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50  
from tensorflow import keras
from tensorflow.autograph import set_verbosity
from sklearn.model_selection import train_test_split
from data_preparation import get_data
import pandas as pd
import matplotlib.pyplot as plt
import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.savefig("data/hystory.png")


data, labels = get_data(600)

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(10, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)
#model.summary()

for layer in baseModel.layers:
	layer.trainable = False

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=Adam(lr=0.001),
              metrics=["accuracy"])

            
history = model.fit(trainX, trainY, epochs=20, batch_size=20, validation_data=(testX, testY))
plot_learning_curves(history)

model.save('model.h5')   
#new_model = tf.keras.models.load_model('model.h5')