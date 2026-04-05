##  2️⃣ env_rl.py (RL Environment)
import gymnasium as gym                 # Use Gymnasium instead of Gym to avoid deprecation warnings.ر
from gymnasium import spaces
import numpy as np

class AdversarialEnv(gym.Env):
    def __init__(self, gan_samples, classifier, max_steps=10):
        super(AdversarialEnv, self).__init__()
        self.gan_samples = gan_samples
        self.classifier = classifier
        self.max_steps = max_steps
        self.num_features = gan_samples.shape[1]
        self.action_space = spaces.Box(low=-0.02, high=0.02, shape=(self.num_features,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_features,), dtype=np.float32)
        self.current_step = 0
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        random_idx = np.random.randint(0, len(self.gan_samples))
        self.state = np.copy(self.gan_samples[random_idx])
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        self.state = np.clip(self.state + action, 0, 1)
        prediction = self.classifier.predict(self.state.reshape(1, -1))[0]
        if prediction == 0: 
            reward = 1.0
            terminated = True 
        else:
            reward = -0.1
            terminated = False
        truncated = self.current_step >= self.max_steps
        return self.state, reward, terminated, truncated, {}
