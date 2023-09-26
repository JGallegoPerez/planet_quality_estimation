import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import shutil
from IPython.display import display, HTML
from IPython.display import Image as IPython_Image


def show_best_worst(image_files, show_paths=True):
    """
    Takes a sorted list of images paths and shows the first 10 (the best) and the last 10 (the worst). 
    """
    # Create a 2x20 subplot grid
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    print("Best images vs. worst images: ")
    # Loop through the first 10 images
    for i in range(10):
        image_path = image_files[i]
        image = Image.open(image_path)
        image = image.resize((128, 128))
        axes[0, i].imshow(image)
        axes[0, i].set_title(f"Good {i+1}")
        axes[0, i].axis('off')
        if show_paths:
            print(image_path)
    # Loop through the last 10 images
    for i in range(10):
        image_path = image_files[-(i+1)]
        image = Image.open(image_path)
        image = image.resize((128, 128))
        axes[1, i].imshow(image)
        axes[1, i].set_title(f"Bad {i+1}")
        axes[1, i].axis('off')
        if show_paths:
            print(image_path)
    plt.show()


def folder_to_data(folder, n=None):
    """"
    Arguments: 
    - "folder" contains planet images 
    - "n": if not None, refers to a number of best and worst pictures selected (to avoid computational overhead). Thus, the dataset will consist of the N best and the N worst images.
    Returns:
    - a sorted 2D numpy array that encodes planet images. Dimensions: (2*N) by number of pixels of an image. 
    - a sorted dictionary that allows a mapping between the images paths and the corresponding numpy arrays.
    - a sorted list of the images (paths)
    """
    images_list = sorted([os.path.join(folder, img) for img in os.listdir(folder)])
    if n is not None:
        images_list = images_list[:n] + images_list[-n:]
    num_samples = len(images_list)
    num_pixels = len(np.array(Image.open(images_list[0])).flatten())
    print("num_samples: ", num_samples)
    print("num_pixels: ", num_pixels)
    # arr2d = np.zeros((num_samples, num_pixels))
    arr2d = np.zeros((num_samples, num_pixels), dtype=np.float16)
    path_arr_dict = {} 
    counter = 1
    for row, path in enumerate(images_list):
        # Open the image using Pillow
        img = Image.open(path)
        # Convert the image to a NumPy array, scaled to 0-1.
        # img_array = np.array(img).flatten() / 255
        img_array = np.array(img, dtype=np.float16).flatten() / 255
        arr2d[row,:] = img_array
        path_arr_dict[path] = img_array
        counter += 1
        if counter % 5000 == 0:
            print("progress: ", counter, '/', num_samples)

    return arr2d, path_arr_dict, images_list


def random_selection(folder_path, num_images):
    """
    Takes a folder path and returns a random, smaller list of images paths
    """
    # Ensure the folder path exists
    if not os.path.exists(folder_path):
        return []
    # Get a list of all image paths in the folder
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.png')]
    # Randomly select N image paths
    random_image_paths = random.sample(image_paths, num_images)
    
    return random_image_paths


def paths_to_array(images_list):
    """
    Converts a paths list into a 2D array, necessary to get prediction probabilities
    """
    num_pixels = len(np.array(Image.open(images_list[0])).flatten())
    num_samples = len(images_list)
    print("num_samples: ", num_samples)
    print("num_pixels: ", num_pixels)
    arr2d = np.zeros((num_samples, num_pixels))
    for row, path in enumerate(images_list):
        # Open the image using Pillow
        img = Image.open(path)
        # Convert the image to a NumPy array, scaled to 0-1.
        img_array = np.array(img).flatten() / 255
        arr2d[row,:] = img_array
    print("Shape: ", arr2d.shape)

    return arr2d


def pipp_qual_parser(paths):
    """
    Takes a list of image paths in pipp format and returns a list (float) of the corresponding quality values
    """
    int_list = []
    for p in paths:
        sub_str1 = p.split('%.png')[0]
        qual_val = sub_str1.split('quality_')[1]
        int_list.append(float(qual_val))
        
    return int_list


def images_compare(paths_list1, list1_name, paths_list2, list2_name, index): # Initialize an index (multiple of 10) to keep track of the displayed images
    """
    Displays a row of the 10 first images of the first model and below another 10 first images from the second model.
    With argument 'index' (multiple of 10) we can start the display further from the beginning. 
    Pressing <Enter> shows the subsequent 10 images from each set.
    Pressing <q> + <Enter> stops the function. 
    """
    while True:
        # Create a 2x20 subplot grid
        fig, axes = plt.subplots(2, 10, figsize=(20, 4))
        print(list1_name, " vs. ", list2_name)
        # Loop through the first 10 images
        for i in range(10):
            image = Image.open(paths_list1[i + index -10])
            image = image.resize((128, 128))
            axes[0, i].imshow(image)
            axes[0, i].set_title(f"{list1_name}: {i + index -9}")
            axes[0, i].axis('off')
        # Loop through the last 10 images
        for i in range(10):
            image = Image.open(paths_list2[i + index - 10])
            image = image.resize((128, 128))
            axes[1, i].imshow(image)
            axes[1, i].set_title(f"{list2_name}: {i + index -9}")
            axes[1, i].axis('off')
        plt.show()
        index += 10
        # Press Enter to display the next set of images
        key = input("Press Enter to inspect the next 10 pairs of images. Press 'q' to quit.")
        if key == 'q':
            break
    

def copy_images(paths, destination_folder, n):
    """
    Copies the n first (best) images that correspond to a paths list into another folder. 
    """
    for i in range(n):
        if i < len(paths):
            image_path = paths[i]
            shutil.copy(image_path, os.path.join(destination_folder, os.path.basename(image_path)))


def display_two_img(image1_path, image1_title, image2_path, image2_title):
    """
    Displays two images, one next to each other.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2)) 
    ax1.imshow(image1_path)
    ax1.set_title(image1_title) 
    ax1.axis('off')  
    ax2.imshow(image2_path)
    ax2.set_title(image2_title)  
    ax2.axis('off') 
    plt.tight_layout()
    plt.show()


# def display_two_img(image1_path, image1_title, image2_path, image2_title):
#     """
#     Displays two images, one next to each other, on a Jupyter notebook.
#     """
#     image1_path = image1_path
#     image2_path = image2_path
#     image1_title = image1_title
#     image2_title = image2_title
#     html_code = f'<div style="display:flex; align-items:center; justify-content:center;">'
#     html_code += f'<div style="text-align:center;">'
#     html_code += f'<img src="{image1_path}" style="width:200px;"><br>'
#     html_code += f'<h2>{image1_title}</h2>'
#     html_code += f'</div>'
#     html_code += f'<div style="text-align:center;">'
#     html_code += f'<img src="{image2_path}" style="width:200px;"><br>'
#     html_code += f'<h2>{image2_title}</h2>'
#     html_code += f'</div>'
#     html_code += f'</div>'
#     display(HTML(html_code))


def empty_folder(folder_path):
    """
    Empties a folder if it has any files or directories.
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)


def bright_normalize(arr2d):
    """
    Makes all images in the 2D array have the same average brightness. 
    """
    overall_average = np.mean(arr2d)
    print("Overall brightness: ", overall_average)
    row_averages = np.mean(arr2d, axis=1, keepdims=True)
    scaled_array = arr2d * (overall_average / row_averages)
    return scaled_array


def loss_accuracy_plot(history):
    """
    Takes as argument the history object from the training with TensorFlow, and plots the corresponding loss and accuracy over the epochs.
    """
    acc = [0.] + history.history['accuracy']
    val_acc = [0.] + history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()    
