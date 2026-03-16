import numpy as np

class AddGaussNoise:
    def __init__(self, std: float | tuple[float, float] = 0.01):
        self.std = std

    def __call__(self, image, **kwargs):
        if isinstance(self.std, tuple):
            std = np.random.uniform(self.std[0], self.std[1])
        else:
            std = self.std
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        return np.clip(image + noise, 0.0, 1.0)