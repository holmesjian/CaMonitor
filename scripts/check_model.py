import tensorflow as tf

# Use the path found above — likely pose_landmark_lite.tflite
interpreter = tf.lite.Interpreter(
    model_path='path/to/pose_landmark_lite.tflite')
interpreter.allocate_tensors()

details = interpreter.get_input_details()
print("Input dtype :", details[0]['dtype'])   # float32 or int8?
print("Input shape :", details[0]['shape'])
