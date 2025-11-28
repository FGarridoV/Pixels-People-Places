from UrbanTool.city import City
import urllib
import glob
import os
import sys

dictio = {'b.png': 'url_back',
          'f.png': 'url_front',
          'sb.png': 'url_side_b',
          'sa.png': 'url_side_a'}

zones_folder = 'zones'
cities = map(lambda f: f.split('/')[-1], glob.glob(f'{zones_folder}/*'))

ready = ["'s-Gravenhage_NL", "'s-Hertogenbosch_NL", "Aa en Hunze_NL", "Aalsmeer_NL", "Aalten_NL",
         "Achtkarspelen_NL", "Alblasserdam_NL", "Albrandswaard_NL", "Alkmaar_NL", "Almelo_NL"]
cities = list(cities)

for c in cities[10:]:
    print(f'starting at {c}')
    # Identifiying zero images files
    image_path = f'zones/{c}/imagedb'
    files = os.popen(f'find "{image_path}" -size 0').read().strip().split('\n')
    zero_imgs = []
    for f in files:
        zero_imgs.append(image_path + '/' + f.split('/')[-1])
    print(f'Identified zero files in {c}')

    city = City(c, log = False, panoids = True)
    for i in zero_imgs:
        if i[-3:] == 'png':
            print(i)
            get_info = lambda f: (f[2], f[3]+f[4]) if len(f) == 5 else (f[2], f[3])
            info = get_info(i.split('_'))
            row = int(info[0])
            column = dictio[info[1]]
            link = city.panoids[column][row]

            img_size = 0
            while img_size == 0:
                urllib.request.urlretrieve(link, i)
                img_size = os.path.getsize(i)
            print(f'Dowloaded the picture {i}')
