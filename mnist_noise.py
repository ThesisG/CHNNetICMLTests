# import tensorflow as tf; 

# print(tf.test.is_gpu_available())
# import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow import random as rd
import numpy as np

# Import other libraries
import numpy as np
from matplotlib import pyplot as plt
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import gzip


# import CHN Layer
from CHNLayer import CHNLayer



# fetch dataset
X, y = fetch_data('mnist', return_X_y=True, local_cache_dir='./Datasets')

# convert to "float32"
X, y = X.astype("float32")/255, y.astype("float32")

def add_noise(images, noise_factor=0.7):
    noisy_images = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    noisy_images = np.clip(noisy_images, 0., 1.)
    return noisy_images
x_noisy = add_noise(X)

# split into train and test
trainX, x_test, trainY, y_test = train_test_split(x_noisy, y, test_size=0.2, random_state=42)

# split into train and validation
x_train, x_val, y_train, y_val = train_test_split(trainX, trainY, test_size=0.2, random_state=42)



# initiate metrics
FNN_trainloss_history = []
FNN_valloss_history = []
FNN_test_accuracy = []
FNN_test_loss = []

CHN_trainloss_history = []
CHN_valloss_history = []
CHN_test_accuracy = []
CHN_test_loss = []



# declare hyperparameters
num_seeds = 3
archs = 3
epochs = 30
batchSize = 128

layers = 2
FNN_Hn = 500
CHN_Hn = 500

learning_rate = 0.00001
optimizer = Adam(learning_rate=learning_rate)

loss = "sparse_categorical_crossentropy"



# train and test arcihtectures
for arch in range(archs):
    print(f"Testing Architecure {arch + 1}")
    # train and test models
    for seed in range(num_seeds):
        print(f"Testing for Seed {seed + 1}")

        np.random.seed(seed)
        rd.set_seed(seed)

        #Create FNN model
        FNN_model = Sequential()

        for _ in range(layers):
            FNN_model.add(Dense(FNN_Hn, activation='relu'))

        FNN_model.add(Dense(10, activation="softmax"))
        FNN_model.compile(optimizer=optimizer,
                    loss=loss,
                    metrics=['accuracy'])

        FNN_model.build(x_train.shape)

        FNN_parameters = np.sum([np.prod(var.get_shape()) for var in FNN_model.trainable_weights])

        #Create CHN model
        CHN_model = Sequential()

        for _ in range(layers):
            CHN_model.add(CHNLayer(CHN_Hn, activation='relu'),)

        CHN_model.add(Dense(10, activation="softmax"))

        CHN_model.compile(optimizer=optimizer,
                    loss=loss,
                    metrics=['accuracy'])

        CHN_model.build(x_train.shape)

        CHN_parameters = np.sum([np.prod(var.get_shape()) for var in CHN_model.trainable_weights])
        
        # train FNN
        print("Training FNN")
        FNN_History = FNN_model.fit(x_train, y_train, epochs = epochs, batch_size = batchSize, validation_data=(x_val, y_val))

        # Evaluate FNN
        test_loss, test_accuracy = FNN_model.evaluate(x_test, y_test)
        
        # store FNN metrics
        FNN_test_accuracy.append(test_accuracy)
        FNN_test_loss.append(test_loss)
        FNN_trainloss_history.append(FNN_History.history['loss'])
        FNN_valloss_history.append(FNN_History.history['val_loss'])

        # train CHN
        print("Training CHNNet")
        CHN_History = CHN_model.fit(x_train, y_train, epochs = epochs, batch_size = batchSize, validation_data=(x_val, y_val))

        # Evaluate CHN
        test_loss, test_accuracy = CHN_model.evaluate(x_test, y_test)
        
        # store CHN metrics
        CHN_test_accuracy.append(test_accuracy)
        CHN_test_loss.append(test_loss)
        CHN_trainloss_history.append(CHN_History.history['loss'])
        CHN_valloss_history.append(CHN_History.history['val_loss'])



    # Measurements
    FNN_accuracy_mean = np.mean(FNN_test_accuracy)
    FNN_accuracy_std = np.std(FNN_test_accuracy)
    FNN_loss_mean = np.mean(FNN_test_loss)
    FNN_loss_std = np.std(FNN_test_loss)

    CHN_accuracy_mean = np.mean(CHN_test_accuracy)
    CHN_accuracy_std = np.std(CHN_test_accuracy)
    CHN_loss_mean = np.mean(CHN_test_loss)
    CHN_loss_std = np.std(CHN_test_loss)



    # store results
    with open(f"mnist_noise/mnist_noiseArch{arch}.txt", "w") as metFile:
        # FNN results
        metFile.write("FNN MODEL\n")
        metFile.write(f"Params: {FNN_parameters}\n")
        metFile.write("ACCURACY\n")
        metFile.write(f"Mean: {FNN_accuracy_mean}\n")
        metFile.write(f"std: {FNN_accuracy_std}\n")
        metFile.write("LOSS\n")
        metFile.write(f"Mean: {FNN_loss_mean}\n")
        metFile.write(f"std: {FNN_loss_std}\n\n")

        # CHN results
        metFile.write("CHN MODEL\n")
        metFile.write(f"Params: {CHN_parameters}\n")
        metFile.write("ACCURACY\n")
        metFile.write(f"Mean: {CHN_accuracy_mean}\n")
        metFile.write(f"std: {CHN_accuracy_std}\n")
        metFile.write("LOSS\n")
        metFile.write(f"Mean: {CHN_loss_mean}\n")
        metFile.write(f"std: {CHN_loss_std}")



    # Generate Graphs
    for seed in range(num_seeds):
        plt.plot(FNN_valloss_history[seed], color="c", linewidth=2)
        plt.plot(CHN_valloss_history[seed], color="r", linewidth=2)
        plt.plot(FNN_trainloss_history[seed], color="c", linewidth=0.5)
        plt.plot(CHN_trainloss_history[seed], color="r", linewidth=0.5)
        plt.title(f"mnist_noise: Architecture {arch}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["FNN"] + ["CHN"])
        plt.savefig(f"mnist_noise/mnist_noiseArch{arch}Seed{seed}.pdf")
        plt.clf()

    layers += 2