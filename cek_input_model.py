import tensorflow as tf

model = tf.keras.models.load_model("models/model_classification.h5")
model.summary()

print("\nInput shape dari model:")
print(model.input_shape)
