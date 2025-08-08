import os
import shutil
import argparse


def organize(input_path: str, output_path: str):
    '''
    Organize the medmnist path with the following structure
    [Medmnist]
        -> [training_set]
            -> [class 1]
                -> image.png
            -> [class 2]
                -> ...
        -> [test_set]
            -> [class 1]
                -> ...
        -> [validation_set]
    :param input_path:
    :param output_path:
    '''

    assert os.path.isdir(input_path), f"The path provided {input_path} is not a folder"

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for medmnist in os.listdir(input_path):
        medmnist_input_path = os.path.join(input_path, medmnist)
        medmnist_output_path = os.path.join(output_path, medmnist)

        if not os.path.isdir(medmnist_output_path):
            os.makedirs(medmnist_output_path)

        for image in os.listdir(medmnist_input_path):
            path_to_input_image = os.path.join(medmnist_input_path, image)
            training_set = os.path.join(medmnist_output_path, "training")
            test_set = os.path.join(medmnist_output_path, "testing")
            validation_set = os.path.join(medmnist_output_path, "validation")


            if not os.path.isdir(training_set):
                os.makedirs(training_set)
            if not os.path.isdir(test_set):
                os.makedirs(test_set)
            if not os.path.isdir(validation_set):
                os.makedirs(validation_set)

            if "chestmnist" not in medmnist:
                # extract the class from the name of the image
                image_class = image.split("_")[1].replace(".png", "")

            else:
                # extract the class
                one_hot_encoded_class = image.split("_")
                del one_hot_encoded_class[0]
                one_hot_encoded_class[-1] = one_hot_encoded_class[-1].replace(".png", "")
                try:
                    one_hot_encoded_class.index("1")
                    image_class = "1"
                except:
                    image_class = "0"

            # create the class folder
            training_set = os.path.join(training_set, f"class_[{image_class}]")
            test_set = os.path.join(test_set, f"class_[{image_class}]")
            validation_set = os.path.join(validation_set, f"class_[{image_class}]")


            if not os.path.isdir(training_set):
                os.makedirs(training_set)
            if not os.path.isdir(test_set):
                os.makedirs(test_set)
            if not os.path.isdir(validation_set):
                os.makedirs(validation_set)

            if "test" in image:
                new_name = image.split("_")[0].replace("test", "")+".png"
                dst = os.path.join(test_set, new_name)
            if "train" in image:
                new_name =  image.split("_")[0].replace("train", "")+".png"
                dst = os.path.join(training_set, new_name)
            if "val" in image:
                new_name =  image.split("_")[0].replace("val", "")+".png"
                dst = os.path.join(validation_set, new_name)

            print(f"[INFO]: moving {input_path} -> {dst} ")
            shutil.move(path_to_input_image, dst)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', required=True, help="Path to the Dataset")
    parser.add_argument('-o', '--output_path', required=True, help='Output path for the processed Dataset')

    args = parser.parse_args()

    organize(args.input_path, args.output_path)







