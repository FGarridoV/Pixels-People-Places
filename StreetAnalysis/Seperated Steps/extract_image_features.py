import numpy as np
from keras.preprocessing.image import load_img 
from keras.applications import resnet  
from keras.models import Model


project_name = 'Delft'
images_folder = r"Z:\data\Delft_NL\imagedb"

resnet_module = resnet
resnet_model = resnet.ResNet152()
model = Model(inputs=resnet_model.inputs, outputs=resnet_model.layers[-2].output)

def extract_features_to_file(image_file: str) -> None:
# Open the image as the desired array size
    image = np.array(load_img(f"{images_folder}\\{image_file}", target_size=(224,224)))
    image = image.reshape(1,224,224,3) 

    # Insert the image array into the model and get the feature vector
    module_image = resnet_module.preprocess_input(image)
    image_features = model.predict(module_image, verbose=0)

    np.save(f"Data\\{project_name}\\image_features\\{image_file}.npy", image_features[0])
