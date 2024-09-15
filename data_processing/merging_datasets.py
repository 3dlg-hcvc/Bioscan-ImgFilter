
import os
# Creates directories for storing processed original images.
def create_output_directories(output_dir):

    # create good and bad image directories
    #bad_images_dir = os.path.join(output_dir, "bad_imgs")
    good_images_dir = os.path.join(output_dir, "good_directory")
    bad_images_dir = os.path.join(output_dir,"bad_directory")
    

    os.makedirs(good_images_dir, exist_ok=True)
    os.makedirs(bad_images_dir, exist_ok=True)
   

    #return bad_images_dir, good_images_dir, cropped_output_dir, unbounded_output_dir, bounded_output_dir
    return good_images_dir, bad_images_dir


