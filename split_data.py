import os
import pandas as pd
import random


def extract_label_bisque(filename):
    if "malignant" in filename:
        return 1
    elif "benign" in filename:
        return 0
    else:
        raise ValueError("Filename does not contain 'malignant' or 'benign'")


def extract_label_brecahad(filename):
    if "apoptosis" in filename:
        return 1
    elif "Case" in filename:
        return 0
    else:
        raise ValueError("File is not from BreCaHAD dataset")


""" Config """
TRAIN_PERCENT = 0.65
VAL_PERCENT = 0.15
FOLDER_PATH = "BreCaHAD/pt_files"
OUTPUT_FILENAME = "BreCaHAD/fold2.csv"
EXTRACT_FUNCTION = extract_label_brecahad


def train_val_test_split(file_names, train_percent=TRAIN_PERCENT, val_percent=VAL_PERCENT):
    random.shuffle(file_names)
    total_files = len(file_names)
    train_end = int(total_files * train_percent)
    val_end = int(total_files * (train_percent + val_percent))

    train_files = file_names[:train_end]
    val_files = file_names[train_end:val_end]
    test_files = file_names[val_end:]

    return train_files, val_files, test_files


def main(folder_path, filename, extract_function):
    """
    :param folder_path: folder with pt files
    :param filename: output file path
    :param extract_function: function to classify each file
    :return: saves csv with fold
    """

    # image file names without extensions
    file_names = [os.path.splitext(file)[0] for file in os.listdir(folder_path)]

    train_files, val_files, test_files = train_val_test_split(file_names)

    train_labels = [extract_function(filename) for filename in train_files]
    val_labels = [extract_function(filename) for filename in val_files]
    test_labels = [extract_function(filename) for filename in test_files]

    val_length = max(len(train_files), len(val_files))
    test_length = max(len(train_files), len(test_files))

    val_files += [''] * (val_length - len(val_files))
    val_labels += [None] * (val_length - len(val_labels))
    test_files += [''] * (test_length - len(test_files))
    test_labels += [None] * (test_length - len(test_labels))

    data = {
        'train': train_files,
        'train_label': train_labels,
        'val': val_files,
        'val_label': val_labels,
        'test': test_files,
        'test_label': test_labels
    }

    df = pd.DataFrame(data)
    df.to_csv(filename)


if __name__ == "__main__":
    main(FOLDER_PATH, OUTPUT_FILENAME, EXTRACT_FUNCTION)
