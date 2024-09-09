import tensorflow as tf
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim

def process_image(image, channels=3):
  image = tf.image.decode_png(image, channels=channels)
  image = tf.image.convert_image_dtype(image, tf.float32)

  return image

def parse_function(paths):
  images = [process_image(tf.io.read_file(path)) for path in paths]
  combined_image = tf.concat(images, axis=-1)
  return combined_image

def display_images(images, isSave, title, image_name):
  image_list = []
  num_imgaes = len(title)

  for i in range(num_imgaes):
    image_list.append(images[:, :, i * 3:i * 3 + 3])

  plt.figure(figsize=(25, 10))
  for i in range(num_imgaes):
    if image_list[i].shape[2] == 0:
      continue 
    plt.subplot(1, num_imgaes, i+1)
    plt.title(title[i])
    plt.imshow(image_list[i])
    plt.axis('off')
  
  if isSave:
    plt.savefig(image_name)
  else : 
    plt.show()
  
  plt.close()
    
def generate_images(model, test_input, tar, image_name, isSave):
  prediction = model(test_input, training=True)
  
  if isSave:
    temp_gen_output = prediction[0].numpy()
    temp_target = tar[0].numpy()
    ssim_score, _ = ssim(temp_gen_output, temp_target, data_range = 2, channel_axis = -1, full=True)

    image = tf.concat([tar[0], prediction[0]], axis=-1).numpy()
    display_images(image, True, ['Target', 'Generated'], './generated_images/' + image_name + '_' + str(ssim_score) + '.png')