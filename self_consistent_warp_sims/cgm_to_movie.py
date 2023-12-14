# With an open CGM file, autmoatically page through the image panes, screenshot,
# and then save the image.

import pyautogui
import re
import imageio
import time
import os

# Specify path to save images frames
path = os.path.join(os.getcwd(), "cgm_frames")


def take_screen_shot(num_frames, coordinates):
    """Function to automatically take screenshot images of cgm frames

     The coordinates provided to pyautogui are done by hand by sizing the cgm
     file and then using pyautogui.position() to get the x,y coordinates as seen
     by pyautogui. Mapping out the screenshot box, the region argument can be
     calculated and fed into the screenshot. The region parameters are
     left, top, width, and height. Thus, we only need the x,y coordinate of
     the top left of the box, the x coordinate of the top right corner, and the y
     coordinate of the bottom left coordinates (x1, y1, x2, y3).
     This returns an image object that is then saved.
     Before, I attempted passing the .png string name to perform this
     process in one line. However, the images were not being saved.
     The while-loop will page through the cgm frames and repeat the process.
     The number of cgm frames can be seen by executing info in while in the
     gist>.

     Parameters:
     -----------
     num_frames: int
         Number of cgm frames to page through and screenshots to take.

    coordinates: tuple
         From these coordinates the region of the screen shot is calculated. The
         first two entries are the x,y position of the top left corner x1,y1.
         The third entry is x of the top right corner x2. The last entry is the
         bottom left corner of the screen y3. Thus coordinate = (x1, y1, x2, y3).
    """

    # From initial coordinates calculate region.
    width = coordinates[2] - coordinates[0]
    height = coordinates[3] - coordinates[1]

    # Create region parameter which is (top, left, width, height)
    region = (coordinates[0], coordinates[1], width, height)

    # Do initial screen shot. This will move the mouse over the gist CGM window
    # and click to be sure the window is active. The mouse is then moved away.
    # Note that y=0 is at the top of the screen and positive is downward.
    cnt = 0
    pyautogui.moveTo(coordinates[0] + width / 2, coordinates[1] + height / 2)
    pyautogui.click()
    pyautogui.moveTo(10, 10)

    im1 = pyautogui.screenshot(region=region)
    im1.save(os.path.join(path, f"screenshot{cnt+1}.png"))
    cnt += 1

    # Loop through pixel intensive frames first with added sleep timer for image
    # to fully load
    while cnt < 240:
        pyautogui.press("f")
        time.sleep(4.15)
        im1 = pyautogui.screenshot(region=region)
        im1.save(os.path.join(path, f"screenshot{cnt+1}.png"))
        cnt += 1

    # The image shouldn't be so intensive so the timer can be decreased.
    while cnt < num_frames:
        pyautogui.press("f")
        time.sleep(1.15)
        im1 = pyautogui.screenshot(region=region)
        im1.save(os.path.join(path, f"screenshot{cnt+1}.png"))
        cnt += 1

    return print("Done")


def get_numeric_suffix(filename):
    """Key function to use in re for grabbing the numerical suffixes."""
    match = re.search(r"screenshot(\d+)\.png", filename)
    if match:
        return int(match.group(1))
    else:
        return float("inf")  # Return infinity for files without a numeric suffix


def sort_images_by_suffix(image_folder):
    """Sort images in ascending numerical order.

    The screenshots are saved as screenshotX.png where X is a digit starting
    from 1. In order to make the beam movie, these images are to be sorted in
    ascending order. This function uses the get_numeric_function to act as a
    key in grabbing and sorting.
    The sorted ascending list is then returned.
    """

    image_files = [
        f
        for f in os.listdir(image_folder)
        if f.startswith("screenshot") and f.endswith(".png")
    ]
    sorted_files = sorted(image_files, key=get_numeric_suffix)
    return sorted_files


def create_movie(image_folder, image_files, fps=5):
    """Create beam movie from png frames.

    The individual frames are found in image_folder and the sorted frames are
    provided by the image_files parameter. The images should be sorted first else
    the movie will be a random selection of frames pieced together.
    """

    writer = imageio.get_writer(os.path.join(image_folder, "beam_movie.mp4"), fps=fps)

    for image in image_files:
        writer.append_data(imageio.imread(os.path.join(image_folder, image)))

    writer.close()

    return "Beam movie created"


cgm_frames = 1394
coords = (118, 142, 779, 795)
take_screen_shot(cgm_frames, coords)

# Create movie
images = sort_images_by_suffix(path)
create_movie(path, images, fps=10)
