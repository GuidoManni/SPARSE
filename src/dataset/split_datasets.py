import os
import argparse
import shutil
from tqdm import tqdm


def create_root_dataset_directory(path_to_splitted_dataset: str):
    if not os.path.exists(path_to_splitted_dataset):
        os.makedirs(path_to_splitted_dataset)

    return path_to_splitted_dataset

def create_supervised_and_unsupervised_directories(path_to_splitted_dataset: str):
    supervised_path = os.path.join(path_to_splitted_dataset, "training_supervised")
    unsupervised_path = os.path.join(path_to_splitted_dataset, "training_unsupervised")
    if not os.path.exists(supervised_path):
        os.makedirs(supervised_path)
    if not os.path.exists(unsupervised_path):
        os.makedirs(unsupervised_path)

    return supervised_path, unsupervised_path


def copy_files_to_new_directory(path_to_files: str, supervised_path: str, unsupervised_path: str, percentage: float):
    # step 1: we get the list of the folders in the training directory
    folders = os.listdir(path_to_files)
    # step 2: we iterate over the folders
    for folder in folders:
        # step 3: we create the new directories
        supervised_folder = os.path.join(supervised_path, folder)
        unsupervised_folder = os.path.join(unsupervised_path, folder)
        if not os.path.exists(supervised_folder):
            os.makedirs(supervised_folder)
        if not os.path.exists(unsupervised_folder):
            os.makedirs(unsupervised_folder)
        # step 4: we get the list of the files in the folder
        files = os.listdir(os.path.join(path_to_files, folder))
        # step 5: we get the number of files to move to the supervised directory
        number_of_files = int(len(files) * percentage)
        # step 6: we move the files to the supervised and unsupervised directories
        for i, file in enumerate(tqdm(files)):
            if i < number_of_files:
                shutil.copy(os.path.join(path_to_files, folder, file), os.path.join(supervised_folder, file))
            else:
                shutil.copy(os.path.join(path_to_files, folder, file), os.path.join(unsupervised_folder, file))




def split_dataset(path_to_dataset: str, saving_path: str, percentage_of_supervised_dataset: float):
    print("[INFO]: creating the new dataset directory")
    # step 1: we create the new directory
    path_to_splitted_dataset = create_root_dataset_directory(saving_path)

    # step 2: we create the supervised and unsupervised directories
    supervised_path, unsupervised_path = create_supervised_and_unsupervised_directories(path_to_splitted_dataset)

    # step 3: we move the training data to the new training directories
    print("[INFO]: copying files to new directories")
    copy_files_to_new_directory(os.path.join(path_to_dataset, "training"), supervised_path, unsupervised_path, percentage_of_supervised_dataset)

    # step 4: we copy the validation and test data to the new dataset directory
    print("[INFO]: copying validation and test data to new directories")
    shutil.copytree(os.path.join(path_to_dataset, "validation"), os.path.join(path_to_splitted_dataset, "validation"))
    shutil.copytree(os.path.join(path_to_dataset, "testing"), os.path.join(path_to_splitted_dataset, "testing"))
    print("[INFO]: dataset splitted successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_dataset", type=str, required=True, help="path to the dataset to split")
    parser.add_argument("--saving_path", default=os.getcwd(), type=str, required=False, help="path to save the splitted dataset")
    parser.add_argument("--percentage_of_supervised_dataset", type=float, required=True, help="percentage of the supervised dataset")
    args = parser.parse_args()

    folder_name = args.path_to_dataset.split("/")[-1] + f"_{str(args.percentage_of_supervised_dataset * 100)}"
    saving_path = os.path.join(args.saving_path, folder_name)

    split_dataset(args.path_to_dataset, saving_path, args.percentage_of_supervised_dataset)




