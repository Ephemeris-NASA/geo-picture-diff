import cv2
import os

before_dir = 'before'
after_dir = 'after'
results_dir = 'results'


def list_files_in_directory(directory_path):
    """
    List all files in a given directory.

    Parameters:
    - directory_path (str): Path to the directory.

    Returns:
    - list: A list of file names in the directory.
    """

    # List all entries in the directory
    all_entries = os.listdir(directory_path)

    # Filter out entries that are directories, keeping only files
    files = [entry for entry in all_entries if os.path.isfile(os.path.join(directory_path, entry))]

    return files


def ensure_directory_exists(directory_path):
    """Ensure the specified directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def calculate_the_difference_save_the_results(before_image, after_image, result_prefix):
    # Compute absolute difference
    difference = cv2.absdiff(before_image, after_image)

    # Thresholding
    ret, thresh = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert one of the original images to color
    color_image1 = cv2.cvtColor(before_image, cv2.COLOR_GRAY2BGR)

    # Draw contours in pink
    result_image = cv2.drawContours(color_image1, contours, -1, (0, 200, 200), 2)

    # Or save to file
    cv2.imwrite(os.path.join(results_dir, f'{result_prefix}_result.tif'), result_image)
    cv2.imwrite(os.path.join(results_dir, f'{result_prefix}_result.png'), result_image)


def process_images(before_dir, after_dir, results_dir):
    """Process images from the specified directories and save the results."""
    ensure_directory_exists(results_dir)

    list_of_before = list_files_in_directory(before_dir)
    list_of_after = list_files_in_directory(after_dir)

    for before in list_of_before:
        before_image_path = os.path.join(before_dir, before)
        before_image = cv2.imread(before_image_path, cv2.IMREAD_GRAYSCALE)

        # Extract the base name without extension
        base_name = os.path.splitext(before)[0]
        after_image_path = os.path.join(after_dir, f'{base_name}_w.tif')

        if os.path.isfile(after_image_path):
            after_image = cv2.imread(after_image_path, cv2.IMREAD_GRAYSCALE)
            result_prefix = os.path.join(results_dir, base_name)
            calculate_the_difference_save_the_results(before_image, after_image, result_prefix)
        else:
            print(f"No relevant after-state image for {before} picture")


if __name__ == '__main__':
    process_images(before_dir, after_dir, results_dir)