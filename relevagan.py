##  1️⃣ relevagan.py (GAN – Layer 1)
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Embedding, Flatten, Multiply, LeakyReLU
# // Use Legacy Adam optimizer to ensure GAN stability in newer versions.
from tensorflow.keras.optimizers.legacy import Adam

class RELEVAGAN:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.latent_dim = 100
        self.num_classes = 2

        # 1. // Define Optimizer (Static learning rate for stability).
        self.optimizer = Adam(0.0002, 0.5)

        # 2.// Build and compile the Discriminator.
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss="binary_crossentropy",
            optimizer=self.optimizer,
            metrics=['accuracy']
        )

        # 3. build Generator
        self.generator = self.build_generator()

        # 4. // Build the Combined Model (Training Generator to fool Discriminator).
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        
        # // Generate synthetic attack samples.
        img = self.generator([noise, label])

        # // Set Discriminator as non-trainable for the combined model.
        self.discriminator.trainable = False

        # // Discriminator evaluates the synthetic (fake) attack.
        validity = self.discriminator([img, label])

        self.combined = Model([noise, label], validity)
        self.combined.compile(loss="binary_crossentropy", optimizer=self.optimizer)

    def build_generator(self):
        # // Build the Generator architecture.
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype="int32")
        
        #// Process Label and concatenate with Noise (Conditional GAN input).
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        model_input = Multiply()([noise, label_embedding])

        model = Sequential([
            Dense(128),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.8),
            Dense(256),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.8),
            Dense(512),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.8),
            Dense(self.input_dim, activation="linear") # linear لبيانات الـ Traffic
        ])

        out = model(model_input)
        return Model([noise, label], out)

    def build_discriminator(self):
        # // Build the Discriminator architecture.
        img = Input(shape=(self.input_dim,))
        label = Input(shape=(1,), dtype="int32")

        #// Merge Label and input Features.
        label_embedding = Flatten()(Embedding(self.num_classes, self.input_dim)(label))
        model_input = Multiply()([img, label_embedding])

        model = Sequential([
            Dense(512),
            LeakyReLU(alpha=0.2),
            Dense(256),
            LeakyReLU(alpha=0.2),
            Dense(1, activation="sigmoid")
        ])

        validity = model(model_input)
        return Model([img, label], validity)

    def train(self, X, y, epochs=100, batch_size=32):
        #// Label arrays for training (Real vs. Fake targets).
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        print(f"[INFO] Starting GAN training for {epochs} epochs...")

        for step in range(epochs):
            # ---------------------
            #  train Discriminator
            # ---------------------
            idx = np.random.randint(0, X.shape[0], batch_size)
            real_imgs, labels = X[idx], y[idx].reshape(-1, 1)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict([noise, labels], verbose=0)

            # // Train on Real and Synthetic data.
            d_loss_real = self.discriminator.train_on_batch([real_imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  train Generator
            # ---------------------
            sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # // Print training status every 10 steps for performance.
            if step % 10 == 0:
                print(f"Step {step} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

        return self
