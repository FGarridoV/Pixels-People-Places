import os
import numpy as np
import multiprocessing
import keras

resnet_module = keras.applications.resnet
resnet_model = keras.applications.resnet.ResNet152()
model = keras.models.Model(inputs=resnet_model.inputs, outputs=resnet_model.layers[-2].output)


def find_unextracted_images(images_folder_path: str, project_name: str) -> list[str]:
    non_extracted_images = []
    existing_extractions = []

    print("Gathering all already-extracted image features")
    for file in os.listdir(f"Intermediate Data/{project_name}/image_features"):
        existing_extractions.append(file[:-4])
    
    print(f"{len(existing_extractions)} image features already present, checking for images without features")
    for filename in os.listdir(images_folder_path):
        if not filename.endswith('_s_a.png') and not filename.endswith('_s_b.png'):
            if filename not in existing_extractions:
                non_extracted_images.append(filename)
    
    print(f"{len(non_extracted_images)} images do not already have a corresponding feature file")
    return non_extracted_images


def extract_features_to_file(image_name: str, images_folder: str, project_name: str) -> None:
# Open the image as the desired array size
    image = np.array(keras.preprocessing.image.load_img(f"{images_folder}/{image_name}",
                                                        target_size=(224,224)))
    image = image.reshape(1,224,224,3) 

    # Insert the image array into the model and get the feature vector
    module_image = resnet_module.preprocess_input(image)
    image_features = model.predict(module_image, verbose=0)

    np.save(f"Intermediate Data/{project_name}/image_features/{image_name}.npy", image_features[0])


def extract_image_features(non_extracted_images: list[str], images_folder: str, project_name: str) -> None:
    """Input model and returns time it takes to create the feature extraction vectors"""

    # loop through each image in the dataset
    print(f"Starting images feature vector extraction for project '{project_name}': {len(non_extracted_images)} images to extract given")
    print(f"Images will be taken from '{images_folder}' and saved to Internal/{project_name}/image_features in the local workspace")
    
    pool = multiprocessing.Pool() 
    pool.map(extract_features_to_file, non_extracted_images)