import os
from glob import glob
from subprocess import Popen, PIPE, CalledProcessError
import pandas as pd
import geopandas as gpd

class Parallel:

    def __init__(self, df, temp_folder, params, script = 'image_detection'):
        self.n_cores = os.cpu_count()
        self.data = df
        self.temp_folder = temp_folder
        if script == 'image_detection':
            self.script = Parallel.template_image_detection.format(*params, temp_folder)
            self.script_filename = self._creates_script_file(script, temp_folder)
        else:
            pass
    
    def _creates_script_file(self, script_name, temp_folder):
        file = open(f"{temp_folder}/temp_{script_name}.py", 'w')
        file.write(self.script)
        file.close()
        return f"{temp_folder}/temp_{script_name}.py"

    def _split_data(self, n):
        elements = len(self.data) // n
        parts = [self.data.iloc[elements*i:elements*(i+1)] for i in range(n-1)]
        parts.append(self.data.iloc[elements*(n-1):])
        return parts
    
    def command_gen(self, df, part):
        dbpath = f"{self.temp_folder}/temp_df_part{part}.geojson"
        df.to_file(dbpath)
        command = f'python \"{self.script_filename}\" \"{dbpath}\" {part}'
        return command

    def paralelize_execution(self, cores = 0):
        if cores == 0:
            cores = self.n_cores
        sub_dfs = self._split_data(cores)
        parts = list(range(len(sub_dfs)))

        commands = []
        for d, p in zip(sub_dfs, parts):
            commands.append(self.command_gen(d,p))
        cmds = " & ".join(commands)

        with Popen(cmds, stdout=PIPE, bufsize=1, universal_newlines=True, shell = True, env=os.environ) as p:
            for b in p.stdout:
                print(b, end='')
            
        if p.returncode != 0:
            raise CalledProcessError(p.returncode, p.args)
        
        l = glob(f'{self.temp_folder}/*.geojson')
        l.sort(key = lambda x: int(x.split('_temp_p')[1].split('.')[0]))
        gdfs = map(gpd.read_file, l)
        gdf = pd.concat(gdfs)
        gdf = gdf.reset_index(drop = True)
        for element in l:
            os.remove(element)
        os.remove(self.script_filename)
        return gdf


    template_image_detection = """
import os
import sys
sys.path.insert(0, 'tools')
import pandas as pd
import warnings
warnings.filterwarnings('ignore', '.*The Shapely GEOS version*', )
import geopandas as gpd
from cv_tools import SSD3

sub_df = sys.argv[1]
part = sys.argv[2]

model = '{}'
imagedb = "{}"
sub_classes = {}

gdf = gpd.read_file(sub_df)
n = len(gdf)

if model == 'ssd3':
    cv_model = SSD3()
        
if sub_classes:
    columns = cv_model.sub_classes['class'].to_list()
else:
    columns = cv_model.classes['class'].to_list()

def _apply_image_counts(row, model, im_folder, n, part, sub_classes = True):
    if row.name%100 == 0:
        print(f'Procession images {{row.name}}/{{n}} in part {{part}}...')
    images = [row['im_side_a'], row['im_front'], row['im_side_b'], row['im_back']]
    if images[0] is None:
        values = [-1]*13 if sub_classes else [-1]*90
        return pd.Series(values)
    im_paths = [f"{{im_folder}}/{{im}}" for im in images]
    df = model.count_classes(im_paths, sub_classes = sub_classes)
    df = df.set_index('class')
    return pd.Series(df.to_dict()['counts'])

gdf[columns] = gdf.apply(lambda row: _apply_image_counts(row, cv_model, imagedb, n, part, sub_classes = sub_classes), axis = 1)
final_json = f"{}/counts_temp_p{{part}}.geojson"
gdf.to_file(final_json)

if os.path.isfile(sub_df):
    os.remove(sub_df)

print(final_json)
    """



