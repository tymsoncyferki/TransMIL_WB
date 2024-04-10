import os
import pandas as pd
import random


def extract_label(filename):
    if "malignant" in filename:
        return 1
    elif "benign" in filename:
        return 0
    else:
        raise ValueError("Filename does not contain 'malignant' or 'benign'")


def train_val_test_split(file_names, train_percent=0.7, val_percent=0.15, test_percent=0.15):
    random.shuffle(file_names)
    total_files = len(file_names)
    train_end = int(total_files * train_percent)
    val_end = int(total_files * (train_percent + val_percent))

    train_files = file_names[:train_end]
    val_files = file_names[train_end:val_end]
    test_files = file_names[val_end:]

    return train_files, val_files, test_files


def main():
    # Path to the folder containing images
    folder_path = "pt_files"

    # Get list of image file names (without extensions)
    file_names = [os.path.splitext(file)[0] for file in os.listdir(folder_path)]

    # Perform train-val-test split
    train_files, val_files, test_files = train_val_test_split(file_names)

    # Extract labels for train, val, and test sets
    train_labels = [extract_label(filename) for filename in train_files]
    val_labels = [extract_label(filename) for filename in val_files]
    test_labels = [extract_label(filename) for filename in test_files]

    # Calculate lengths of val and test sets
    val_length = max(len(train_files), len(val_files))
    test_length = max(len(train_files), len(test_files))

    # Append blank values to val and test sets to make them the same length
    val_files += [''] * (val_length - len(val_files))
    val_labels += [None] * (val_length - len(val_labels))

    test_files += [''] * (test_length - len(test_files))
    test_labels += [None] * (test_length - len(test_labels))

    # Create DataFrame
    data = {
        'train': train_files,
        'train_label': train_labels,
        'val': val_files,
        'val_label': val_labels,
        'test': test_files,
        'test_label': test_labels
    }

    df = pd.DataFrame(data)

    # Save DataFrame to CSV
    df.to_csv('fold0.csv')


if __name__ == "__main__":
    main()