import sys
import json
import pandas as pd
from zone import Zone

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
    if len(municipalities.loc[municipalities['executed'] == 0]) > 0:
        return True
    else:
        return False


if __name__ == '__main__':

    if len(sys.argv) == 2:

        list_csv = sys.argv[1]
        while is_next_city(list_csv):
            municipalities = pd.read_csv(list_csv)
            # Next city to run
            m = municipalities.loc[municipalities['executed'] == 0].iloc[0]
            municipalities.at[m.name, 'executed'] = 2
            municipalities.to_csv(list_csv, index = False)

            #Start execution
        
            name = m['name']
            region = m['region']
            country = m['country']
            country_code = m['country_code']
            logs = True if m['logs'] else False

            n_tags = number_of_tags(list_csv)
            tags = []
            for t in range(1,n_tags+1):
                tag_aux = {'name': m[f'tag:name{t}'], 
                           'kind': m[f'tag:kind{t}'], 
                           'tag': load_tag(m[f'tag:tag{t}'])}
                tags.append(tag_aux)
                
            boundary_opt = m['boundary_opt']
            network_type = m['network_type']
            street_types = m['street_types']
            gap_grid = m['gap_grid']
            zone_grid = m['zone_grid']

            download_images = True if m['download_images'] else False
            city_accuracy = True if m['city_accuracy'] else False

            try:   
                print(f'Starting collection from {name}...')

                ### Creates the zone class
                zone = Zone(name, region, country, country_code, logs = logs)

                ### Creates the zones folders
                zone.create_zones_folders()

                ### Set boundary
                zone.set_boundary(boundary_opt, accuracy = city_accuracy)

                ### Set network
                zone.set_network(network_type, street_types, simplify = True)

                ### Add geometries
                for t in tags:
                    zone.new_geometry(t['name'], t['tag'], t['kind'], and_tags = None)

                ### Set grid points
                zone.set_grid_points(gap = gap_grid, key_crop = zone_grid)

                ### Get panoids from gridpoints
                res1 = zone.get_panoids(user = False)
                if res1 == -1:
                    zone.logger.write_log(f'{name} Collection failed (panoid collection)')
                    municipalities = pd.read_csv(list_csv)
                    municipalities.at[m.name, 'executed'] = -2
                    municipalities.to_csv(list_csv, index = False)
                    print(f'Going to next city')
                    continue

                ### Get collects GSV
                res2 = zone.collect_street_view_images(url_only = not download_images)
                if res2 == -1:
                    zone.logger.write_log(f'{name} Collection failed (images collection)')
                    municipalities = pd.read_csv(list_csv)
                    municipalities.at[m.name, 'executed'] = -2
                    municipalities.to_csv(list_csv, index = False)
                    print(f'Going to next city')
                    continue 

                if download_images:
                    zone.logger.write_log(f'{name} Execution successfully, checking the downloads')
                    try:
                        zone.check_downloads()
                        zone.logger.write_log(f'{name} downloads checked')
                    except:
                        zone.logger.write_log(f'{name} downloaded couldnt be checked')

                zone.logger.write_log(f'{name} Collection runs successfully')
                municipalities = pd.read_csv(list_csv)
                municipalities.at[m.name, 'executed'] = 1
                municipalities.to_csv(list_csv, index = False)


            except Exception as e:
                zone.logger.write_log(f'{name} collection failed')
                zone.logger.write_error(e)
                municipalities = pd.read_csv(list_csv)
                municipalities.at[m.name, 'executed'] = -1
                municipalities.to_csv(list_csv, index = False)
                print(f'Going to next city')
        print('CSV completed')

    else:
        print(f"Error: You should give the csv list")


   




