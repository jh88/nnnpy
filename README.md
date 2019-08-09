# Feedforward neural network with NumPy

Dependencies: SciPy, NumPy and tqdm

```
from nnnpy.models import Model
from nnnpy.layers import Dense, Dropout, BatchNormalization, Activation
from nnnpy.optimizers import Adam


model = Model()
model.add(Dense(449, input_dim=128, weights_initializer='glorot_normal'))
model.add(Activation('tanh'))
model.add(Dropout(0.707))

model.add(Dense(741, weights_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.953))

model.add(Dense(10, activation='softmax', weights_initializer='he_uniform'))
opt = Adam(lr=0.0001)

model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

model.fit(data_x, data_y, batch_size=512, epochs=195)
```
