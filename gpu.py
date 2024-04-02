import tensorflow as tf

print("""


 _____ ______  _   _   _____  _____  _____  _____ 
|  __ \| ___ \| | | | |_   _||  ___|/  ___||_   _|
| |  \/| |_/ /| | | |   | |  | |__  \ `--.   | |  
| | __ |  __/ | | | |   | |  |  __|  `--. \  | |  
| |_\ \| |    | |_| |   | |  | |___ /\__/ /  | |  
 \____/\_|     \___/    \_/  \____/ \____/   \_/  
                                                  
                                                  


""")

print('tensorflow version',tf.__version__)

# https://www.tensorflow.org/api_docs/python/tf/test/is_gpu_available
gpu_available = tf.test.is_gpu_available()
is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
is_cuda_gpu_min_3 = tf.test.is_gpu_available(True, (3,0))

print("gpu_available = ", gpu_available)
print("is_cuda_gpu_available = ", is_cuda_gpu_available)
print("is_cuda_gpu_min_3 = ", is_cuda_gpu_min_3)
