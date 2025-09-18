from clearml import Dataset

# Step 1: Create a new dataset (this registers it with your ClearML server)
dataset = Dataset.create(
    dataset_name="airports_malls",
    dataset_project="FLoc",
    dataset_version="1.0.1",  # Optional; auto-increments if omitted
    # output_uri="s3://my-bucket/datasets/",  #c  Specify your storage (e.g., S3, GS, Azure, or /mnt/share/)
    description="fixed ray ordering to left-right"
)

# Step 2: Add local files or folders
dataset.add_files(
    path="/home/kevinmeng/workspace/mappedin/VPS/f3loc/data",  # Local path to your data
    #wildcard="*.jpg",  # Optional: Filter files (e.g., images only)
    recursive=True,  # Include subfolders
    # dataset_path="/data/"  # Optional: Path within the dataset
)

# Step 3: Upload files to the specified storage
dataset.upload(
    show_progress=True,  # Display progress bar
    # output_url="s3://my-bucket/datasets/",  # Override if needed
    compression=None  # Optional: Use ZIP_DEFLATED for compression
)

# Step 4: Finalize the dataset (commits changes and updates metadata on ClearML server)
dataset.finalize()

# Step 5: Publish (optional, makes it visible/public in ClearML UI)
# dataset.publish()