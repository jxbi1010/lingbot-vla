import pyarrow.parquet as pq
from pathlib import Path
import numpy as np

# Update this to your data directory
DATA_DIR = Path("/home/ss-oss1/data/user/shawks/DATA/agilex-real-world-data/disk-1/Badminton_shuttlecock_storage")

def inspect_parquet():
    # Find the first parquet file to use as a sample
    parquet_files = sorted(list(DATA_DIR.glob("**/*.parquet")))
    
    if not parquet_files:
        print(f"No parquet files found in {DATA_DIR}")
        return

    sample_file = parquet_files[0]
    print(f"Inspecting sample file: {sample_file}\n")
    print(f"{'Key':<40} | {'Arrow Type':<20} | {'Python Type'} | {'Shape/Len'}")
    print("-" * 100)

    # Read the table
    table = pq.read_table(sample_file)
    # Convert first row to a dictionary for easy inspection
    first_row = table.to_batches()[0].to_pydict()

    for column_name in table.column_names:
        field = table.schema.field(column_name)
        arrow_type = str(field.type)
        
        # Get an example value
        example_val = first_row[column_name][0]
        python_type = type(example_val).__name__
        
        # Determine shape or length if applicable
        shape_info = "N/A"
        if isinstance(example_val, (list, np.ndarray)):
            shape_info = len(example_val)
        elif isinstance(example_val, dict):
            shape_info = f"Keys: {list(example_val.keys())}"

        # Flag problematic "struct" types (often pointers to videos)
        warning = " <!! STRUCT !!" if "struct" in arrow_type else ""

        print(f"{column_name:<40} | {arrow_type:<20} | {python_type:<12} | {shape_info}{warning}")

if __name__ == "__main__":
    inspect_parquet()