import os
import tensorflow as tf
import util
import model
import time

#### Read datasets
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

BUFFER_SIZE = 400
BATCH_SIZE = 1
TEST_PATH = './datasets/trainData/'
CHECKPOINT_DIR = './training_checkpoints/'

color_1layer_files = tf.data.Dataset.list_files(str(TEST_PATH + 'color_1layer/*.png'), shuffle=False)
depth_1layer_files = tf.data.Dataset.list_files(str(TEST_PATH + 'depth_1layer/*.png'), shuffle=False)
motion_vector_1layer_files = tf.data.Dataset.list_files(str(TEST_PATH + 'motion_vector_1layer/*.png'), shuffle=False)

color_2layer_files = tf.data.Dataset.list_files(str(TEST_PATH + 'color_2layer/*.png'), shuffle=False)
depth_2layer_files = tf.data.Dataset.list_files(str(TEST_PATH + 'depth_2layer/*.png'), shuffle=False)
motion_vector_2layer_files = tf.data.Dataset.list_files(str(TEST_PATH + 'motion_vector_2layer/*.png'), shuffle=False)

motion_blur_files = tf.data.Dataset.list_files(str(TEST_PATH + 'motion_blur/*.png'), shuffle=False) #target

test_dataset =  tf.data.Dataset.zip((color_1layer_files, depth_1layer_files, motion_vector_1layer_files, 
                                              color_2layer_files, depth_2layer_files, motion_vector_2layer_files, motion_blur_files))

test_dataset = test_dataset.map(lambda color_1layer, depth_1layer, motion_vector_1layer, color_2layer, depth_2layer, motion_vector_2layer, motion_blur: 
                                                  util.parse_function([color_1layer, depth_1layer, motion_vector_1layer, color_2layer, depth_2layer, motion_vector_2layer, motion_blur]), 
                                                  num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

#Init model
generator = model.Generator()

#Load weight
checkpoint = tf.train.Checkpoint(generator=generator)
checkpoint.restore(os.path.join(CHECKPOINT_DIR, "ckpt")).expect_partial()

for test_batch in test_dataset.take(10):
  inp = test_batch[:, :, :, :-3]
  tar = test_batch[:, :, :, 18:]
  
  start = time.time()
  util.generate_images(generator, inp, tar, 'Trained', True)

  print(f'Time taken for generating: {time.time() - start:.2f} sec\n')