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
    if not 'processed' in municipalities.columns:
        municipalities['processed'] = 0
        municipalities.to_csv(list_csv, index = False)

    cond1 = (municipalities['processed'] == 0)
    cond2 = (municipalities['executed'] == 1)

    if len(municipalities.loc[cond1 & cond2]) > 0:
        return True
    else:
        return False


if __name__ == '__main__':

    if len(sys.argv) == 2:

        list_csv = sys.argv[1]
        while is_next_city(list_csv):
            municipalities = pd.read_csv(list_csv)

            # Next city to run
            cond1 = (municipalities['processed'] == 0)
            cond2 = (municipalities['executed'] == 1)
            m = municipalities.loc[cond1 & cond2].iloc[0]
            municipalities.at[m.name, 'processed'] = 2
            municipalities.to_csv(list_csv, index = False)

            #Start execution
            name = m['name']
            region = m['region']
            country = m['country']
            country_code = m['country_code']

            try:   
                print(f'Starting image processing for {name}...')

                folder = f'{name}_{country_code}'
                city = City(folder, utm = False, log_code = "vp", panoids = True)

                if not city.downloaded_images_were_checked():
                    city.logger.write_log(f'{name} checking downloaded images')
                    city.check_downloads()
                
                city.detect_objects()

                city.logger.write_log(f'{name} process runs successfully')
                municipalities = pd.read_csv(list_csv)
                municipalities.at[m.name, 'processed'] = 1
                municipalities.to_csv(list_csv, index = False)

            except Exception as e:
                city.logger.write_log(f'{name} image processing failed')
                city.logger.write_error(e)

                municipalities = pd.read_csv(list_csv)
                municipalities.at[m.name, 'processed'] = -1
                municipalities.to_csv(list_csv, index = False)
                print(f'Going to next city')
        print('CSV completed')

    else:
        print(f"Error: You should give the csv list")


   




