import urllib
import os
import time

dictio = {'b.png': 'url_back',
          'f.png': 'url_front',
          'sb.png': 'url_side_b',
          'sa.png': 'url_side_a'}

def check_and_solve(city_folder, panoids):

    global dictio
    print(f'starting at {city_folder}')
    # Identifiying zero images files
    image_path = f'zones/{city_folder}/imagedb'
    proc = os.popen(f'find "{image_path}" -size 0')
    files = proc.read().strip().split('\n')
    proc.close()
    zero_imgs = []
    for f in files:
        zero_imgs.append(image_path + '/' + f.split('/')[-1])
    print(f'Identified zero files in {city_folder}')

    for i in zero_imgs:
        if i[-3:] == 'png':
            print(i)
            get_info = lambda f: (f[2], f[3]+f[4]) if len(f) == 5 else (f[2], f[3])
            info = get_info(i.split('_'))
            row = int(info[0])
            column = dictio[info[1]]
            link = panoids[column][row]

            img_size = 0
            while img_size == 0:
                try:
                    urllib.request.urlretrieve(link, i)
                    img_size = os.path.getsize(i)
                except:
                    print("sleeping 10 seconds...")
                    time.sleep(10)
                    img_size = 0
            print(f'Dowloaded the picture {i}')
    
    f = open(f'zones/{city_folder}/checked', 'w')
    f.close()

