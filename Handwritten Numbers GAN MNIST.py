import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU, Dropout # Import Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle

# Define constants
IMG_WIDTH, IMG_HEIGHT = 28, 28
IMG_CHANNELS = 1
LATENT_DIM = 100

# Load MNIST dataset
(X_train, _), (_, _) = mnist.load_data()

# Rescale and normalize pixel values
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=3)

# Define the generator model
def build_generator():
    model = Sequential()
    model.add(Dense(7*7*128, input_dim=LATENT_DIM))
    model.add(Reshape((7, 7, 128)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh'))
    return model

# Define the discriminator model
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# Build the generator and discriminator models
generator = build_generator()
discriminator = build_discriminator()

# Compile the discriminator model
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy']) # Add accuracy metric

# Define the combined generator and discriminator model
z = Input(shape=(LATENT_DIM,))
img = generator(z)
discriminator.trainable = False
validity = discriminator(img)
combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Train the GAN
for epoch in range(10):
    # Sample random noise from a normal distribution
    noise = np.random.normal(0, 1, (32, LATENT_DIM))
    
    # Generate a batch of new images
    gen_imgs = generator.predict(noise)
    
    # Select a random batch of images from the training set
    idx = np.random.randint(0, X_train.shape[0], 32)
    imgs = X_train[idx]
    
    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(imgs, np.ones((32, 1))) # Train on real images
    d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((32, 1))) # Train on fake images
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) # Average the losses

    # Train the generator
    g_loss = combined.train_on_batch(noise, np.ones((32, 1)))
    
    # Print the progress
    # Access the first element of g_loss, which should represent the loss value.
    print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0])) # Changed g_loss to g_loss[0]

# Use the trained generator to create new logos
noise = np.random.normal(0, 1, (25, LATENT_DIM))
gen_imgs = generator.predict(noise)

# Display the generated logos
plt.figure(figsize=(5, 5))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(gen_imgs[i, :, :, 0], cmap='gray')
    plt.axis('off')
plt.show()
