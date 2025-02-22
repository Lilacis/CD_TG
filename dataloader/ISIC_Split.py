import pandas as pd
import os
from PIL import Image

df = pd.read_csv('.../isic/class_id.csv')

dir = '.../dataset/ISIC/ISIC2018_Task1-2_Training_Input/'

dict = {'seborrheic_keratosis' : '3','nevus' : '2','melanoma' : '1'}

for directory in ['1', '2', '3']:
    subdir = os.path.join(dir,directory)
    if not os.path.exists(subdir):
        os.makedirs(subdir)

for idx, row in df.iterrows():

    src_path = dir + row['ID'] + '.jpg'
    dest_path = dir + dict[row['Class']] + '/' + row['ID'] + '.jpg'

    try:
        img = Image.open(src_path)
        img_resized = img.resize((512, 512))
        img_resized.save(dest_path)
    except:
        continue
