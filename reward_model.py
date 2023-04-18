from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.optimizers import Adam
# import tensorflow as tf

class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.q = Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        q = self.q(x)
        return q