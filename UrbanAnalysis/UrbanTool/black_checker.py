import glob
import cv2
import concurrent.futures as cf
from skimage.exposure import is_low_contrast

def is_black_image(filename, threshold = 0.05):
    gray = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
    if is_low_contrast(gray,threshold):
        print(f"Black image found: {filename}")
        return 1
    else:
        return 0

if __name__ == '__main__':

    zones_folder = "zones"
    zones = glob.glob(f"{zones_folder}/*")

    writer = open('black_images.csv', 'w')
    writer.write('municipality, black_images')
    writer.close()

    for z in zones:
        if f"{z}/checked" in glob.glob(f"{z}/*"):
            municipality = z.split("/")[-1]
            if municipality != "'s-Hertogenbosch_NL":
                images = glob.glob(f"{z}/imagedb/*")
                print(f"Starting with: {municipality}")
                with cf.ThreadPoolExecutor() as executor:
                    results = executor.map(is_black_image, images)
                black_images = sum(results)

                writer = open('black_images.csv', 'a')
                writer.write('\n')
                writer.write(f"{municipality},{black_images}")
                writer.close()
                print(f"{municipality} ready with {black_images} black images")
