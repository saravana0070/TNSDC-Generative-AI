import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Flatten, Reshape, Input
from tensorflow.keras.optimizers import Adam

# Define the generator model
def build_generator():
    generator = Sequential([
        Dense(256, input_dim=100),
        LeakyReLU(0.2),
        BatchNormalization(),
        Dense(512),
        LeakyReLU(0.2),
        BatchNormalization(),
        Dense(1024),
        LeakyReLU(0.2),
        BatchNormalization(),
        Dense(1156, activation='tanh')  # Output dimension matches the flattened image size (34*34)
    ])
    noise = Input(shape=(100,))
    img = generator(noise)
    img = Reshape((34, 34))(img)  # Reshape to match the image size
    return Model(noise, img)

# Define the discriminator model
def build_discriminator():
    discriminator = Sequential([
        Flatten(input_shape=(34, 34)),  # Flatten the input image
        Dense(512),
        LeakyReLU(0.2),
        Dense(256),
        LeakyReLU(0.2),
        Dense(1, activation='sigmoid')
    ])
    img = Input(shape=(34, 34))
    validity = discriminator(img)
    return Model(img, validity)

# Load your handwritten alphabet dataset and preprocess it
def load_dataset():
    data = []
    labels = []
    folder_path = 'alphabets'
    categories = os.listdir(folder_path)  # Make sure this only contains your 27 folders
    categories.sort()  # Sort the categories if needed

    for index, category in enumerate(categories):
        subfolder_path = os.path.join(folder_path, category)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, filename)
                if os.path.isfile(image_path):
                    image = Image.open(image_path).convert('L')  # Convert to grayscale
                    image = image.resize((34, 34))  # Resize to match input size
                    image = np.array(image)
                    # Normalize the pixel values to the range [-1, 1]
                    image = (image.astype(np.float32) - 127.5) / 127.5
                    data.append(image)
                    labels.append(index)  # Assuming each folder's index is its label

    return np.array(data), np.array(labels)

# Load your handwritten alphabet dataset
X_train, y_train = load_dataset()

# Compile both models
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

# Combined model
z = Input(shape=(100,))
img = generator(z)
discriminator.trainable = False
validity = discriminator(img)
combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

# Training parameters
batch_size = 128
epochs = 10000
sample_interval = 1000

# Training loop
for epoch in range(epochs):
    # Train discriminator
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    imgs = X_train[idx]
    noise = np.random.normal(0, 1, (batch_size, 100))
    gen_imgs = generator.predict(noise)
    d_loss_real = discriminator.train_on_batch(imgs, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    valid_y = np.array([1] * batch_size)
    g_loss = combined.train_on_batch(noise, valid_y)

    # Print progress
    if epoch % sample_interval == 0:
        print(f"Epoch {epoch}, [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%], [G loss: {g_loss}]")

        # Generate sample images
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        plt.show()
