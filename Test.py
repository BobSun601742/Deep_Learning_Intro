import sys
print(sys.executable)
print(sys.version)

try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
except Exception as e:
    print("TensorFlow is not installed.")
    print("Error:", e)