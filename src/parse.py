import os
import pandas as pd
from xml.dom import minidom
import xml.etree.ElementTree as ET



def parser(path=None):
    files = os.listdir(path)
    files = [file for file in files if file.endswith('.xml')]
    df = []
    for file in files:
        temp_path = os.path.join(path, file)
        tree = ET.parse(temp_path)
        root = tree.getroot()
        for elem in root:
            for subelem in elem:
                li = subelem.attrib['points'].split()

            y = [val.split(',')[0] for val in li]
            x = [val.split(',')[1] for val in li]
            xmin = min(x)
            ymin = min(y)
            xmax = max(x)
            ymax = max(y)
            temp = {'filename' : file.split('.')[0],
                    'xmin' : xmin,
                    'ymin' : ymin,
                    'xmax' : xmax,
                    'ymax' : ymax}
            df.append(temp)
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(path, 'mask.csv'), index=False)


path = r'E:\source\sem_8\invoice\invoice\Annotations'
parser(path)

