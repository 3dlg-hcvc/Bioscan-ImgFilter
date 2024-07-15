import os
import json

# Load image annotations 
def load_annotations(input_dir, file_name):
    with open(os.path.join(input_dir, file_name)) as f:
        return json.load(f)
    

# Save updated annotations back to JSON file.
def save_annotations(annotations,output_dir, file_name):
    with open(os.path.join(output_dir, file_name), "w") as f:
        json.dump(annotations, f, indent=4)