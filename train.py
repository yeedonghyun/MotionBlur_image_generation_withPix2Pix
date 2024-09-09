import time
import datetime
import os
import tensorflow as tf
import util
import model

from matplotlib import pyplot as plt
from IPython import display

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

IS_TEST = False
IS_LOGGING = True
BUFFER_SIZE = 400
BATCH_SIZE = 1
TOTAL_STEPS = 100000
TEST_PATH = './datasets/trainData/'
TRAIN_PATH = './datasets/trainData/'
TITLE = ['color 1layer image', 'depth 1layer image', 'motion vector 1layer image', 
         'color 2layer image', 'depth 2layer image', 'motion vector 2layer image', 'motion blur image']

#train
color_1layer_files = tf.data.Dataset.list_files(str(TRAIN_PATH + 'color_1layer/*.png'), shuffle=False)
depth_1layer_files = tf.data.Dataset.list_files(str(TRAIN_PATH + 'depth_1layer/*.png'), shuffle=False)
motion_vector_1layer_files = tf.data.Dataset.list_files(str(TRAIN_PATH + 'motion_vector_1layer/*.png'), shuffle=False)

color_2layer_files = tf.data.Dataset.list_files(str(TRAIN_PATH + 'color_2layer/*.png'), shuffle=False)
depth_2layer_files = tf.data.Dataset.list_files(str(TRAIN_PATH + 'depth_2layer/*.png'), shuffle=False)
motion_vector_2layer_files = tf.data.Dataset.list_files(str(TRAIN_PATH + 'motion_vector_2layer/*.png'), shuffle=False)

motion_blur_files = tf.data.Dataset.list_files(str(TRAIN_PATH + 'motion_blur/*.png'), shuffle=False) #target

train_dataset =  tf.data.Dataset.zip((color_1layer_files, depth_1layer_files, motion_vector_1layer_files, 
                                              color_2layer_files, depth_2layer_files, motion_vector_2layer_files, motion_blur_files))

train_dataset = train_dataset.map(lambda color_1layer, depth_1layer, motion_vector_1layer, color_2layer, depth_2layer, motion_vector_2layer, motion_blur: 
                                                  util.parse_function([color_1layer, depth_1layer, motion_vector_1layer, color_2layer, depth_2layer, motion_vector_2layer, motion_blur]), 
                                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

#test
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

#### Test datasets
if IS_TEST :
  for batch in train_dataset.take(1):
    util.display_images(batch[0].numpy(), False, TITLE, 'test')

  for batch in train_dataset.take(1):
    print (f'Discriminator input image shape: {batch.shape}')
    print (f'Generator input image shape: {batch[:, :, :, :-3].shape}')
    print (f'target image shape: {batch[:, :, :, 18:].shape}')

  for batch in train_dataset.take(1):
    image = tf.expand_dims(batch[0].numpy()[:, :, :], 0)
    down_model = model.downsample(32, 4)
    down_result = down_model(image, 0)
    print (f'Init image shape: {batch.shape} to down result shape: {down_result.shape}')
    
    up_model = model.upsample(18, 4)
    up_result = up_model(down_result)
    print (f'Down result shape: {down_result.shape} to up result shape: {up_result.shape}')

#### Generator and discriminator test
generator = model.Generator()
discriminator = model.Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

if IS_TEST :
  for batch in train_dataset.take(1):
    input = batch[0].numpy()[:, :, :-3]
    gen_output = generator(input[tf.newaxis, ...], training=False)
    plt.imshow(gen_output[0, ...])
    plt.show()

    disc_out = discriminator([input[tf.newaxis, ...], gen_output], training=False)
    plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
    plt.colorbar()
    plt.show()

    target = batch[:, :, :, 18:]
    disc_out = discriminator([input[tf.newaxis, ...], target], training=False)
    plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
    plt.colorbar()
    plt.show()

#### Logging
if IS_LOGGING : 
  checkpoint_dir = './training_checkpoints'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                  discriminator_optimizer=discriminator_optimizer,
                                  generator=generator,
                                  discriminator=discriminator)

  log_dir="logs/"
  summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

#### Training
@tf.function
def train_step(input_images, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_images[tf.newaxis, ...], training=True)

    disc_real_output = discriminator([input_images[tf.newaxis, ...], target], training=True)
    disc_generated_output = discriminator([input_images[tf.newaxis, ...], gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = model.generator_loss(disc_generated_output, gen_output, target)
    disc_loss = model.discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

def fit(train_ds, test_ds, steps):
  start = time.time()
  for example_batch in test_ds.take(1):
    example_input = example_batch[:, :, :, :-3]
    example_target = example_batch[:, :, :, 18:]
  
  for step, images in train_ds.repeat().take(steps).enumerate():
    if (step) % 1000 == 0:
      display.clear_output(wait=True)
      util.generate_images(generator, example_input, example_target, 'Training_' + str(round(time.time() - start)), True)
      print(f"Step: {step//1000}k")

    input_images = images[0].numpy()[:, :, :-3]
    target = images[:, :, :, 18:]
    train_step(input_images, target, step)

    # Training step
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)

    # Save (checkpoint) the model every 5k steps
    if (step + 1) % 5000 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)
    
    step += 1
  
  print(f'Time taken for training: {time.time()-start:.2f} sec\n')

fit(train_dataset, test_dataset, TOTAL_STEPS)

if IS_LOGGING : 
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))