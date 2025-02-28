import os
from PIL import Image
from multiprocessing import Pool, cpu_count

# Set your main dataset folder (contains Train, Validation, Test)
main_folder = "KaggleData"
output_folder = "RGBAKaggleData"

os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

def convert_image(file_info):
    """Convert image to RGBA (if necessary) and save it in the correct format."""
    input_path, output_path = file_info

    try:
        with Image.open(input_path) as img:
            # Convert to RGBA for formats like PNG
            if img.mode != 'RGBA' and img.format != 'JPEG':
                img = img.convert("RGBA")  # Convert to RGBA if not JPEG
            else:
                # Convert to RGB if saving as JPEG (since JPEG doesn't support alpha channel)
                if img.format == 'JPEG':
                    img = img.convert("RGB")
            
            # Ensure subfolder structure exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path)  # Save with the same format
        return f"Converted: {input_path} → {output_path}"
    
    except Exception as e:
        return f"Error with {input_path}: {e}"

def get_image_files(root_dir, output_root):
    """Recursively get all image files in root_dir, mapping them to output_root."""
    image_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'webp')):
                input_path = os.path.join(subdir, file)
                rel_path = os.path.relpath(input_path, root_dir)  # Preserve subfolder structure
                output_path = os.path.join(output_root, rel_path)
                image_files.append((input_path, output_path))
    return image_files

# Get all image file paths from Train, Validation, and Test directories
image_files = get_image_files(main_folder, output_folder)

# Use multiprocessing to process images faster
if __name__ == '__main__':
    with Pool(cpu_count()) as p:
        results = p.map(convert_image, image_files)

    # Print conversion results
    for res in results:
        print(res)
