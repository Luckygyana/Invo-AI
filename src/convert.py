
import os
import lxml
from PIL import Image
import pandas as pd

file_path = r'E:\source\sem_8\invoice\invoice\annotate\pdf_imgs'
files = os.listdir(file_path)
list_csv = [file for file in files if file.endswith('.csv')]
list_imgs = [file for file in files if file.endswith('.jpg')]

for path in list_imgs:
    path = os.path.join(file_path, path)
    filename = path.split('\\')[-1]
    img = Image.open(path)
    height, width = img.size
    df = pd.read_csv(path.split('.')[0]+'.csv')
    df['height'] = height
    df['width'] = width
    df['filename'] = filename
    df.rename(columns = {'filename' : 'NAME_ID', 'xmin' : 'XMIN', 'ymin' : 'YMIN', 'width' : 'W', 'height' : 'H', 'xmax' : 'XMAX', 'ymax' : 'YMAX', 'label' : 'Label'}, inplace=True)
    print(df.columns)
    df.to_csv(path.split('.')[0]+'.csv', index=False, columns= ['NAME_ID', 'XMIN', 'YMIN', 'W', 'H', 'XMAX', 'YMAX', 'Label'])


def produceOneCSV(list_of_files, file_out):
   # Consolidate all CSV files into one object
   result_obj = pd.concat([pd.read_csv(os.path.join(file_path, file)) for file in list_of_files])
   # Convert the above object into a csv file and export
   result_obj.to_csv(file_out, index=False, encoding="utf-8")

out_path = r'E:\source\sem_8\invoice\invoice\annotate\pdf_imgs\out.csv'
print(list_csv)
produceOneCSV(list_csv, out_path)