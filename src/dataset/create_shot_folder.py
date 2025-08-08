import os
import shutil
import random
import argparse

def move_images(dataset_path, output_path, n_supervised):
    supervised_path = os.path.join(output_path, 'training_supervised')
    unsupervised_path = os.path.join(output_path, 'training_unsupervised')
    test_path = os.path.join(output_path, "testing")
    val_path = os.path.join(output_path, "validation")

    # Ensure the supervised directory exists
    if not os.path.exists(supervised_path):
        os.makedirs(supervised_path)

    if not os.path.exists(unsupervised_path):
        os.makedirs(unsupervised_path)

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    if not os.path.exists(val_path):
        os.makedirs(val_path)

    subfolders = os.listdir(dataset_path)
    for folder in subfolders:
        path_to_folder = os.path.join(dataset_path, folder)

        if folder == "testing" or folder == "validation":
            for image_class in os.listdir(path_to_folder):
                path_to_classes = os.path.join(path_to_folder, image_class)
                for images in os.listdir(path_to_classes):
                    path_to_images = os.path.join(path_to_classes, images)
                    output_path_to_classes = os.path.join(output_path, folder, image_class)
                    os.makedirs(output_path_to_classes, exist_ok=True)
                    dst = os.path.join(output_path_to_classes, images)
                    print(f"[INFO]: copying {path_to_images} -> {dst}")
                    shutil.copy(path_to_images, dst)
        if folder == "training":
            for image_class in os.listdir(path_to_folder):
                path_to_classes = os.path.join(path_to_folder, image_class)
                images = os.listdir(path_to_classes)
                unsupervised_images = images.copy()

                # sample the supervised images
                supervised_images = random.sample(images, min(n_supervised, len(images)))

                # remove the images that have been selected for the supervision
                for image_to_remove in supervised_images:
                    unsupervised_images.remove(image_to_remove)

                # now copy
                for supervised_image in supervised_images:
                    path_to_images = os.path.join(path_to_classes, supervised_image)
                    output_path_to_classes = os.path.join(supervised_path, image_class)
                    os.makedirs(output_path_to_classes, exist_ok=True)
                    dst = os.path.join(output_path_to_classes, supervised_image)
                    print(f"[INFO]: copying {path_to_images} -> {dst}")
                    shutil.copy(path_to_images, dst)

                for unsupervised_image in unsupervised_images:
                    path_to_images = os.path.join(path_to_classes, unsupervised_image)
                    output_path_to_classes = os.path.join(unsupervised_path, image_class)
                    os.makedirs(output_path_to_classes, exist_ok=True)
                    dst = os.path.join(output_path_to_classes, unsupervised_image)
                    print(f"[INFO]: copying {path_to_images} -> {dst}")
                    shutil.copy(path_to_images, dst)

    print("Image moving process completed.")

# Example usage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', required=True, help="Path to the Dataset")
    parser.add_argument('-o', '--output_path', required=True, help='Output path for the processed Dataset')

    args = parser.parse_args()
    n_supervised = [5, 10, 20, 50]

    os.makedirs(args.output_path, exist_ok=True)

    for n in n_supervised:
        medmnists = os.listdir(args.input_path)

        for medmnist in medmnists:
            print(f"[INFO]: Processing {medmnist}")
            os.makedirs(os.path.join(args.output_path, medmnist+f"_{n}shot"), exist_ok=True)
            move_images(os.path.join(args.input_path, medmnist), os.path.join(args.output_path, medmnist+f"_{n}shot"), n)
