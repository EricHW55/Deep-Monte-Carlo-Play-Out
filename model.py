import tensorflow as tf
from tensorflow import keras as keras
# from keras.backend import relu, sigmoid
# from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Model
# from keras.optimizers import Adam

# tf.debugging.set_log_device_placement(True)

class Model(tf.keras.Model):
    def __init__(self):
        super(Model,self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.mae = tf.keras.losses.MeanAbsoluteError()

    def action_model(self):
        input_ = tf.keras.layers.Input(shape=(7,))
        self.flatten = tf.keras.layers.Flatten()(input_)
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')(self.flatten)
        self.fc2 = tf.keras.layers.Dense(7, activation='softmax')(self.fc1)
        model = tf.keras.models.Model(input_, self.fc2)
        # model.compile()
        return model
    
    def reward_model(self):
        input_ = tf.keras.layers.Input(shape=(7,))
        self.flatten = tf.keras.layers.Flatten()(input_)
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')(self.flatten)
        self.fc2 = tf.keras.layers.Dense(1, activation='sigmoid')(self.fc1)
        model = tf.keras.models.Model(input_, self.fc2)
        # model.compile()
        return model

    def action_train(self, model, x, y, **kwargs):
        epochs = kwargs['epochs']
        action = kwargs['action']
        for _ in range(epochs):
            with tf.GradientTape() as tape:
                tape.watch(model.trainable_weights)
                prediction = model(x)[-1]
                # print('Prediction : {}'.format(prediction))
                loss = self.mse([y], [prediction[action]])
            
            # print('Loss : {}'.format(loss))
            gradients = tape.gradient(loss, model.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    
    def reward_train(self, model, x, y, **kwargs):
        epochs = kwargs['epochs']
        print(y)
        for _ in range(epochs):
            with tf.GradientTape() as tape:
                tape.watch(model.trainable_weights)
                prediction = model(x)
                print('Prediction : {}'.format(prediction))
                # loss = self.mse(y, prediction)
                loss = self.mae(y, prediction)
                
            
            print('Loss : {}'.format(loss))
            gradients = tape.gradient(loss, model.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            # print(model(data)[-1][4])

    #     # self.inp = tf.keras.layers.Input(shape=(6,7))
    #     # self.cnv1 = tf.keras.layers.Conv2D(64, kernel_size=(2,2), padding='SAME', activation='relu', input_shape=(6,))
    #     # self.mp1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='SAME')
    #     self.flatten = tf.keras.layers.Flatten(input_shape=(6,7))
    #     self.fc1 = tf.keras.layers.Dense(128, activation='relu')
    #     self.fc2 = tf.keras.layers.Dense(7) # , activation='softmax'
    #     # self.flatten = tf.keras.layers.Flatten(input_shape=(6,7))
    #     # self.fc1 = tf.keras.layers.Dense(512, activation='relu')
    #     # self.fc2 = tf.keras.layers.Dense(256, activation='relu')
    #     # # self.dropout = tf.keras.layers.Dropout(0.2)
    #     # self.fc3 = tf.keras.layers.Dense(7, activation='softmax')
        
    # def call(self, inputs):
    #     # x = self.inp()(inputs)
    #     # x = self.cnv1(inputs)
    #     # x = self.mp1(x)
    #     x = self.flatten(inputs)
    #     x = self.fc1(x)
    #     x = self.fc2(x)
    #     # x = self.flatten(inputs)
    #     # x = self.fc1(x)
    #     # x = self.fc2(x)
    #     # # x = self.dropout(x)
    #     # x = self.fc3(x)
    #     return x


if __name__ == "__main__":
    m = Model()
    from reward_model import DQN
    reward_model = DQN()

    # m = Model()
    # model = m.action_model()
    # reward_model = m.reward_model()
    # model.summary()
    data = tf.convert_to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0], dtype=tf.int32)
    data = tf.reshape(data, (1,42))
    # print(data)
    # results = round(113273796/114046769, 2)
    reward = 113274560/114049194
    # print('results :', results)
    # print('reward :', reward)
    # # m.action_train(model, data, results, epochs=8, action=4)
    # # print(model(data)[-1][4])
    m.reward_train(reward_model, data, reward, epochs=1)
    print(float(reward_model(data)))


    # data = tf.convert_to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0], dtype=tf.int32)
    # data = tf.reshape(data, (6,7))
    # reward = 107744966/114049157
    # print(reward)
    # print(reward_model(data)[-1][0])
    # m.reward_train(reward_model, data, reward, epochs=8)
    # print(reward_model(data)[-1][0])


    # data = tf.convert_to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1], dtype=tf.int32)
    # data = tf.reshape(data, (6,7))
    # reward = 113654370/114049179
    # print(reward)
    # print(reward_model(data)[-1][0])
    # m.reward_train(reward_model, data, reward, epochs=8)
    # print(reward_model(data)[-1][0])

   
    # data = tf.convert_to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 1, 0, -1, -1], dtype=tf.int32)
    # data = tf.reshape(data, (6,7))
    # reward = 2326995/2327283
    # print(reward)
    # print(reward_model(data)[-1][0])