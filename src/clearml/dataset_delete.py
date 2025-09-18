from clearml import Dataset

# Get the dataset
dataset = Dataset.get(
                      dataset_id="9c346fa3887246d6bf509a1a57fdd577")

if dataset:
    # Delete from ClearML server (metadata only)
    dataset.delete()
    print("Dataset deleted from ClearML server.")
else:
    print("Dataset not found.")