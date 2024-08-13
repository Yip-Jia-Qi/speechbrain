import os
import random
import csv

# Set the random seed for reproducibility
random.seed(42)

# Define data directories
data_dir = "/data3/jiaqi.yip/musdb18_stems/8k"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Get list of training and testing files
trainset = os.listdir(train_dir)
testset = os.listdir(test_dir)

# Select 10 random items from trainset for validation set
validationset = random.sample(trainset, 10)

# Remove the validation items from the training set
trainset = [item for item in trainset if item not in validationset]

# Create CSV files for each set
def create_csv(song_list, data_dir, dataset, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID","song_path"])
        counter = 0
        for item in song_list:
            song_path = os.path.join(data_dir, dataset, item)  # Adjust path based on data structure
            writer.writerow([counter, song_path])
            counter += 1

# Create CSV files
create_csv(trainset, data_dir, "train", "train_data.csv")
create_csv(validationset, data_dir, "train", "validation_data.csv")
create_csv(testset, data_dir, "test", "test_data.csv")

print("CSV files created successfully!")
