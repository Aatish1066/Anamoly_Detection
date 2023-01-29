import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset into a pandas dataframe
df = pd.read_csv('venv/datasets/NSL-KDD.txt')

# Encode the categorical variables to numerical values
encoder = LabelEncoder()
df['protocol_type'] = encoder.fit_transform(df['protocol_type'])
df['service'] = encoder.fit_transform(df['service'])
df['flag'] = encoder.fit_transform(df['flag'])
df['output'] = encoder.fit_transform(df['output'])

# Split the dataset into training and testing sets
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(50, activation='relu', input_shape=(X_train.shape[1],)),
  tf.keras.layers.Dense(30, activation='relu'),
  tf.keras.layers.Dense(22, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=11, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print('Test Accuracy:', test_acc)
model.save('model.h5')
