# Pixels · People · Places: Computer Vision and Image Embeddings for Perception-Aware Urban Analytics

Author: F.O. Garrido-Valenzuela (important part of this repository was contributed by Max Lange)
CityAI Lab
Transport and Logistic Group
Faculty of Technology, Policy, and Management
Delft University of Technology
Corresponding author: F.O. Garrido-Valenzuela

#### Contact Information:
f.garridov@uc.cl
Delft University of Technology - Faculty of Technology, Policy, and Management
Jaffalaan 5
2628 BX Delft
The Netherlands

### General Introduction
This repository contains:
- Scripts for collecting different layers of urban data systematically.
- Scripts for performing different urban analysis. Regressions, Correlations, Embeddings generations, among other. 
- Post-processed data required for some analysis (processed) data.

The codes were used to conduct different studies at Delft University
of Technology, as part of Francisco Garrido-Valenzuela's PhD Thesis project (2021-2025):

This code and data is being made public both to act as supplementary data for publications and the PhD
thesis of Francisco Garrido-Valenzuela and in order for other researchers to use this data in their own
work.

### Purpose of the Codes
These codes operazionalizes the data collection and analysis of the studies. Based on just indicating the name of city, or its geographical boundaries. These scripts collect the different layers of information, analyze the data, for doing different urban analysis. 

### How to collect data and analyze different urban correlations/inferences
This repository is structured in two main folders
- UrbanAnalysis
    - UrbanTool: This is a self contained script for collecting urban data. As input you need a city in the format `city, country` as text. For instance, `Delft, the Netherlands`. By providing that information (or a list of cities as indicated in zone_lists folders), this scripts will conduct a data collection in the city specified. Specifically, this code retrieve: 
        - Point of interests (POI) from Open Street Map (OSM) 
        - Land use data from Open Street Map (OSM)
        - Street network from Open Street Map (OSM)
        - Traffic Junctions from Open Street Map (OSM)
        - Image ID (Panorama IDs) from Google Street View (GSV). For retrieving the actual images, a Google API key is needed.
    For executing the functionalities of these scripts you can use the following two commands:
        - ```python run_list_data_collection.py zone_lists/list.csv``` -> This will collect the data.
        - ```python run_list_image_processing.py zone_lists/list.csv```-> This will process the images (be aware that images needs to be deleted afterwards)
    Once finished, all the information is stored in a folder with: City boundaries, POIs, land uses, image_db, and panoids
    - Notebooks: These are different playgrounds used to analyze the data and produce the reported results.
    - Aggregated Results: These are the standarized betas we found per Municipality.
    - Extra Data sources: Some additional spatial layers.
- StreetAnalysis
    - complete_run.ipynb: This is the core notebook. This notebook summaries the full process of starting with the image embeddings for Delft, to producing street embeddings.
    - Data: Intermediate data (in spatial files)
    - Figures: Different plots and figures generated in the process (complete_run)
    - Scripts and Separated_steps: Codes for implementing each section individually. `complete_run` consume part of these scripts.

## License CC BY-NC-SA 4.0

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg







