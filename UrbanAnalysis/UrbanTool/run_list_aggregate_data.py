# Run list data aggregation
import sys
import os
import json
import pandas as pd
from zone import Zone
from UrbanTool.city import City

def number_of_tags(list_csv):
    df = pd.read_csv(list_csv)
    columns = df.columns
    nums = [int(t[-1]) if 'tag' in t else -1 for t in columns] 
    return max(nums)


def load_tag(tag_dict):
    tag = json.loads(tag_dict)
    for k in tag:
        tag[k] = True if tag[k].capitalize() == 'True' else tag[k]
    return tag


def is_next_city(list_csv):
    municipalities = pd.read_csv(list_csv)
    
    # Add processed column if not exist
    if not 'aggregated' in municipalities.columns:
        municipalities['aggregated'] = 0
        municipalities.to_csv(list_csv, index = False)

    cond1 = (municipalities['aggregated'] == 0)
    cond2 = (municipalities['processed'] == 1)
    cond3 = (municipalities['executed'] == 1)

    if len(municipalities.loc[cond1 & cond2 & cond3]) > 0:
        return True
    else:
        return False


if __name__ == '__main__':

    if len(sys.argv) == 2:

        list_csv = sys.argv[1]
        year = (-1,-1)
        while is_next_city(list_csv):
            municipalities = pd.read_csv(list_csv)
            
            # Next city to run
            cond1 = (municipalities['aggregated'] == 0)
            cond2 = (municipalities['processed'] == 1)
            cond3 = (municipalities['executed'] == 1)

            m = municipalities.loc[cond1 & cond2 & cond3].iloc[0]
            municipalities.at[m.name, 'aggregated'] = 2
            municipalities.to_csv(list_csv, index = False)

            #Start execution
            name = m['name']
            region = m['region']
            country = m['country']
            agg_cell = m['aggregation_cell']
            country_code = m['country_code']

            try:   
                print(f'Starting data aggregation for {name}...')

                folder = f'{name}_{country_code}'
                city = City(folder, utm = True, log_code = 'ap', boundary = True,
                                                                edges = True,
                                                                nodes = True,
                                                                geometries = True,
                                                                detections = True)

                min_year = city.detections_utm['year'].min() if year[0] == -1 else year[0]
                max_year = city.detections_utm['year'].max() if year[1] == -1 else year[1]

                city.aggregate_info_in_cells(min_year, max_year, [(city.amenities_utm, 'amenity')], [(city.landuse_utm, 'landuse')], side = agg_cell)

                city.logger.write_log(f'{name} process runs successfully')
                municipalities = pd.read_csv(list_csv)
                municipalities.at[m.name, 'aggregated'] = 1
                municipalities.to_csv(list_csv, index = False)

            except Exception as e:
                city.logger.write_log(f'{name} data aggregation failed')
                city.logger.write_error(e)

                municipalities = pd.read_csv(list_csv)
                municipalities.at[m.name, 'aggregated'] = -1
                municipalities.to_csv(list_csv, index = False)
                print(f'Going to next city')
        print('CSV completed')

    else:
        print(f"Error: You should give the csv list")


   




