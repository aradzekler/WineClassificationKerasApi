import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models
from tensorflow.keras import layers


wine_data = load_wine()  # and loading it.
feature_data = wine_data['data']

label_data = wine_data['target']

X_train, X_test, y_train, y_test = train_test_split(feature_data, label_data
                                                    , test_size=0.3)

scaler = MinMaxScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# a nice way to create a feature column
feature_cols = [tf.feature_column.numeric_column('x', shape=[13])]


# Sequential model because we are going to have plenty of consecutive
# and connected layers.
dnn_keras_model = models.Sequential()
dnn_keras_model.add(layers.Dense(units=13, input_dim=13, activation='relu'))

# no need to state input_dim, keras knows to link the layers together.
dnn_keras_model.add(layers.Dense(units=13, activation='relu'))
dnn_keras_model.add(layers.Dense(units=13, activation='relu'))
dnn_keras_model.add(layers.Dense(units=3, activation='softmax'))

# compile all the layers together.
dnn_keras_model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
# if the data is NOT 1HOT encoded, use sparse_categorical_entropy

# start training!
dnn_keras_model.fit(scaled_X_train, y_train, epochs=50)

# our testing predictions
predictions = dnn_keras_model.predict_classes(scaled_X_test)

print(classification_report(predictions, y_test))