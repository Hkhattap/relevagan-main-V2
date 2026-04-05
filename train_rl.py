## 3️⃣ train_rl.py (PPO)
import numpy as np
from stable_baselines3 import PPO
from env_rl import AdversarialEnv # // Verify names match correctly in this step.

def train_rl(gan_model, classifier, X_train_scaled, num_samples=500):
    # // Generate synthetic GAN samples for training.
    latent_dim = 100
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    attack_labels = np.ones((num_samples, 1))
    gan_attacks = gan_model.generator.predict([noise, attack_labels], verbose=0)

    # // Initialize the Environment.
    env = AdversarialEnv(gan_samples=gan_attacks, classifier=classifier)

    # // Start Model Training.
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=1000) # عدد قليل للسرعة في البداية

    return model
