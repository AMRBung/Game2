# dqn_agent_model.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from utils import CombineStreams  # Импортируем пользовательский слой из utils.py


# Создание модели Dueling DQN
def build_dueling_dqn(input_shape_param, num_actions):
    inputs = Input(shape=input_shape_param)
    x = Conv2D(32, (8, 8), strides=4, activation="relu")(inputs)
    x = Conv2D(64, (4, 4), strides=2, activation="relu")(x)
    x = Conv2D(64, (3, 3), strides=1, activation="relu")(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)

    # Dueling Architecture
    value = Dense(1, activation="linear")(x)
    advantage = Dense(num_actions, activation="linear")(x)

    # Создаем объект кастомного слоя CombineStreams
    combine_layer = CombineStreams()
    q_values = combine_layer([value, advantage])  # Применяем слой

    model = Model(inputs=inputs, outputs=q_values)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="huber_loss"
    )
    return model


# Пример использования
if __name__ == "__main__":
    input_shape = (84, 84, 4)  # Входные данные - четыре фрейма для истории
    n_actions = 5  # Количество возможных действий
    dqn_model = build_dueling_dqn(input_shape, n_actions)
    dqn_model.summary()  # Отобразить структуру модели
