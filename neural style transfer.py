import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import PIL.Image
import time

# Function to convert a tensor to a PIL image.
# It scales the tensor values to the 0-255 range, converts it to a numpy array, and then to a PIL image.
def tensor_to_pil_image(tens):
    tens = tens * 255
    tens = np.array(tens, dtype=np.uint8)
    if np.ndim(tens) > 3:
        assert tens.shape[0] == 1
        tens = tens[0]
    return PIL.Image.fromarray(tens)

# Function to compute the Gram matrix.
# This is used in style transfer to capture the style of an image by computing the matrix using the einsum function and normalizing it.
def compute_gram_matrix(tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    tensor_shape = tf.shape(tensor)
    num_locations = tf.cast(tensor_shape[1] * tensor_shape[2], tf.float32)
    return result / num_locations

# Function to load and preprocess an image.
# It reads the image file, decodes it, converts it to float32 type, resizes it while maintaining the aspect ratio, and adds a batch dimension.
def load_and_preprocess_image(image_path):
    max_dim = 256
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# Function to display an image using Matplotlib.
# It optionally sets a title, removes the axis, and shows the image.
def display_image(img, img_title=None, fig_size=(12, 12)):
    plt.figure(figsize=fig_size)
    if len(img.shape) > 3:
        img = tf.squeeze(img, axis=0)
    plt.imshow(img)
    if img_title:
        plt.title(img_title)
    plt.axis('off')
    plt.show()
    plt.close()

# Function to create a VGG model that returns a list of intermediate output values from the specified layers.
def vgg_intermediate_layers(layer_list):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_list]
    model = tf.keras.Model([vgg.input], outputs)
    return model

# Class defining a custom model that extracts style and content features from specified layers of the VGG model.
# It computes the Gram matrix for style layers and extracts the content directly from content layers.
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_intermediate_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_outputs = [compute_gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}

# Function to calculate the style and content loss.
# It computes the mean squared error between the current outputs and the target outputs for both style and content, and then combines these losses with weights.
def compute_style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight, num_style_layers, num_content_layers):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2) for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    total_loss = style_loss + content_loss
    return total_loss

# Function to perform one optimization step on the image.
# It computes the gradients of the loss with respect to the image and updates the image using these gradients.
@tf.function()
def optimize_image_step(image, extractor, optimizer, style_targets, content_targets, style_weight, content_weight, num_style_layers, num_content_layers, tv_weight):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = compute_style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight, num_style_layers, num_content_layers)
        loss += tv_weight * tf.image.total_variation(image)
    gradients = tape.gradient(loss, image)
    optimizer.apply_gradients([(gradients, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))
    return loss

# Load and preprocess the content and style images.
content_image_path = r"C:\Users\ssrin\Downloads\pexels-pixabay-147411.jpg"
style_image_path = r"C:\Users\ssrin\Downloads\starry night image.jpg"
content_image = load_and_preprocess_image(content_image_path)
style_image = load_and_preprocess_image(style_image_path)

# Display the content and style images.
display_image(content_image, 'Content Image', fig_size=(6, 6))
display_image(style_image, 'Style Image', fig_size=(6, 6))

# Define the VGG model and specify the layers to be used for style and content extraction.
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# Extract style features from the style image using the specified layers.
style_feature_extractor = vgg_intermediate_layers(style_layers)
style_outputs = style_feature_extractor(style_image * 255.0)

# Print the details of the extracted style features.
for name, output in zip(style_layers, style_outputs):
    print(name)
    print("  shape: ", output.numpy().shape)
    print("  min: ", output.numpy().min())
    print("  max: ", output.numpy().max())
    print("  mean: ", output.numpy().mean())
    print()

# Create an instance of the custom model to extract style and content features from the specified layers.
extractor = StyleContentModel(style_layers, content_layers)
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

# Initialize the image to be optimized with the content image.
optimized_image = tf.Variable(content_image)

# Define the optimizer and loss weights.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
style_loss_weight = 1e-2
content_loss_weight = 1e4
tv_loss_weight = 30

# Train the model for a specified number of epochs and steps per epoch.
start_time = time.time()
epochs_count = 10
steps_per_epoch_count = 100

for epoch in range(epochs_count):
    for step in range(steps_per_epoch_count):
        optimize_image_step(optimized_image, extractor, optimizer, style_targets, content_targets, style_loss_weight, content_loss_weight, num_style_layers, num_content_layers, tv_loss_weight)
        print('.', end='')
    print('Train step: {}'.format(epoch * steps_per_epoch_count + step + 1))

end_time = time.time()
print("Total time: {:.1f}".format(end_time - start_time))

# Display the resulting image.
display_image(optimized_image.numpy()[0], 'Resulting Image', fig_size=(6, 6))

# Convert the final optimized image to a PIL image and save it.
final_image = tensor_to_pil_image(optimized_image)
final_image.save('result.png')