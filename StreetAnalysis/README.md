
# Project description
This project aims to develop a deep-learning model for street classification using street-view images as input. The model should be able to generate an embedding (n-dimensional vector) for posterior classification through a clustering method. Finally, the categories of streets are spatially visualised.

# Process
All steps conducted in this project and the related notebooks are explained here.
## 1. Select a pre-trained Convolutional Neural Network
After examining different models and seeing which fits the project's purpose the most, a pre-trained convolutional neural network is selected.  
:arrow_right: testing_pretrained_CNN.ipynb
## 2. Feature extraction
Using the selected pre-trained CNN, a feature vector from each image.  
:arrow_right: Pretrained_CNN_Extraction.ipynb
## 3. Averaging feature vector per geometry unit
Assigning images to the geometry unit and averaging the feature vectors in the same geometry unit  
:arrow_right: Assigning_Image_to_Geometry.ipynb  
:arrow_right: Merging_Features.ipynb
## 4. Clustering
Apply clustering methods to the feature vector dataset.  
:arrow_right: Kmeans_clustering.ipynb  
:arrow_right: kmeans_hr_gm.ipynb
## 5. Interpreting and labelling clusters
Explore the clustering results and label them.  
:arrow_right: Assign_labels_clusters.ipynb
## 6. Visualisation of clusters
Finally visualising the clustering results.  
:arrow_right: Results_Visualisation_Kepler.ipynb -- Using kepler_gl library  
:arrow_right: results_visualisation.ipynb -- Using Vega-altair






