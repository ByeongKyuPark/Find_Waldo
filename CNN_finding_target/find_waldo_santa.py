import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import tensorflow as tf
import time

# directories
image_dir = os.path.join(os.getcwd(), 'images', 'finding_waldo')
background_dir = os.path.join(image_dir, 'background.jpg')
waldo_dir = os.path.join(image_dir, 'waldo.png')
santa_dir = os.path.join(image_dir, 'santa.png')  

# load images
background_im = Image.open(background_dir)
waldo_im = Image.open(waldo_dir).resize((60, 100))
santa_im = Image.open(santa_dir).resize((60, 100))  

# generate random sample images
def generate_sample_image():
    background_im_resized = background_im.resize((500, 350))
    waldo_im_resized = waldo_im
    
    col = np.random.randint(0, 410)
    row = np.random.randint(0, 230)
    rand_person = np.random.choice([0, 1], p=[0.5, 0.5])
    
    if rand_person == 1:
        background_im_resized.paste(waldo_im_resized, (col, row), mask=waldo_im_resized)
        cat = 'Waldo'
    else:
        background_im_resized.paste(santa_im, (col, row), mask=santa_im)
        cat = 'Santa'
        
    return np.array(background_im_resized).astype('uint8'), (col, row), rand_person, cat

# draw bounding box
def plot_bounding_box(image, gt_coords, pred_coords=None):
    image = Image.fromarray(image)    
    draw = ImageDraw.Draw(image)
    draw.rectangle((gt_coords[0], gt_coords[1], gt_coords[0] + 60, gt_coords[1] + 100), outline='green', width=5)
    if pred_coords:
        draw.rectangle((pred_coords[0], pred_coords[1], pred_coords[0] + 60, pred_coords[1] + 100), outline='red', width=5)
    return image

# data generation
def generate_data(batch_size=16):
    while True:
        x_batch = np.zeros((batch_size, 350, 500, 3))
        y_batch = np.zeros((batch_size, 1))
        boundary_box = np.zeros((batch_size, 2))
        
        for i in range(batch_size):
            sample_im, pos, person, _ = generate_sample_image()
            x_batch[i] = sample_im / 255
            y_batch[i] = person
            boundary_box[i, 0] = pos[0]
            boundary_box[i, 1] = pos[1]
            
        yield {'input_layer': x_batch}, {'class': y_batch, 'box': boundary_box}  

# model architecture
def convolutional_block(inputs):
    x = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 6, padding='valid', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 6, padding='valid', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    return x

def regression_block(x):
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(2, name='box')(x)
    return x

def classification_block(x):
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', name='class')(x)
    return x

# generate the model
inputs = tf.keras.Input((350, 500, 3))
x = convolutional_block(inputs)
box_output = regression_block(x)
class_output = classification_block(x)
model = tf.keras.Model(inputs=inputs, outputs=[class_output, box_output])

# callbacks and learning rate scheduler
class VisCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 2 == 0:
            test_model()

def lr_schedule(epoch, lr):
    if (epoch + 1) % 5 == 0:
        lr *= 0.2
    return max(lr, 3e-7)

# compile and train the model
model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss={'class': 'binary_crossentropy', 'box': 'mse'}, 
              metrics={'class': 'accuracy', 'box': 'mse'})

# visualize the predictions after epochs
def test_model():
    output_dir = os.path.join(os.getcwd(), 'output_images')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(3):
        sample_im, pos, _, cat = generate_sample_image()
        sample_image_normalized = sample_im.reshape(1, 350, 500, 3) / 255
        predicted_class, predicted_box = model.predict(sample_image_normalized)
        
        # determine the predicted category
        if predicted_class[0][0] > 0.5:
            predicted_category = 'Waldo'
            certainty = predicted_class[0][0]  # Probability of being Waldo
        else:
            predicted_category = 'Santa'
            certainty = 1 - predicted_class[0][0]  # Probability of being Santa
        
        # draw bounding box with color intensity based on certainty
        im = Image.fromarray(sample_im)
        draw = ImageDraw.Draw(im)
        if predicted_category == 'Santa':
            outline_color = (0, 0, int(255 * certainty))  # Blue intensity based on certainty
        else:
            outline_color = (int(255 * certainty), 0, 0)  # Red intensity based on certainty
        draw.rectangle((int(predicted_box[0][0]), int(predicted_box[0][1]), 
                        int(predicted_box[0][0]) + 60, int(predicted_box[0][1]) + 100),
                       outline=outline_color, width=5)
        
        # save the image
        im.save(os.path.join(output_dir, f'image_{i+1}.jpg'))
        print(f"Image {i+1} saved.")
 
# fit the model
hist = model.fit(generate_data(), epochs=16, steps_per_epoch=120, 
                 callbacks=[VisCallback(), tf.keras.callbacks.LearningRateScheduler(lr_schedule)])
