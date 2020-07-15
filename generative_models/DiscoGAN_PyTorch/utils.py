import os
import cv2
import numpy as np
import pandas as pd

celebA_path = './celeba/'

def shuffle_data(da, db):
    a_idx = list(range(len(da)))
    np.random.shuffle( a_idx )

    b_idx = list(range(len(db)))
    np.random.shuffle(b_idx)

    shuffled_da = np.array(da)[ np.array(a_idx) ]
    shuffled_db = np.array(db)[ np.array(b_idx) ]

    return shuffled_da, shuffled_db

def read_images( filenames, domain=None, image_size=64):

    images = []
    for fn in filenames:
        image = cv2.imread(fn)
        if image is None:
            continue

        if domain == 'A':
            kernel = np.ones((3,3), np.uint8)
            image = image[:, :256, :]
            image = 255. - image
            image = cv2.dilate( image, kernel, iterations=1 )
            image = 255. - image
        elif domain == 'B':
            image = image[:, 256:, :]

        image = cv2.resize(image, (image_size,image_size))
        image = image.astype(np.float32) / 255.
        image = image.transpose(2,0,1)
        images.append( image )

    images = np.stack( images )
    return images

def read_attr_file( attr_path, image_dir ):
    f = open( attr_path )
    lines = f.readlines()
    lines = map(lambda line: line.strip(), lines)
    lines=list(lines)
    columns = ['image_path'] + lines[1].split()
    lines = lines[2:]

    items = map(lambda line: line.split(), lines)
    df = pd.DataFrame( items, columns=columns )
    df['image_path'] = df['image_path'].map( lambda x: os.path.join( image_dir, x ) )

    return df

def get_celebA_files(style_A, style_B, constraint, constraint_type, test=False, n_test=200):
    attr_file = os.path.join( celebA_path, 'list_attr_celeba.txt' )
    image_dir = os.path.join( celebA_path, 'img_align_celeba' )
    image_data = read_attr_file( attr_file, image_dir )

    if constraint:
        image_data = image_data[ image_data[constraint] == constraint_type]

    style_A_data = image_data[ image_data[style_A] == '1']['image_path'].values
    if style_B:
        style_B_data = image_data[ image_data[style_B] == '1']['image_path'].values
    else:
        style_B_data = image_data[ image_data[style_A] == '-1']['image_path'].values

    if test == False:
        return style_A_data[:-n_test], style_B_data[:-n_test]
    if test == True:
        return style_A_data[-n_test:], style_B_data[-n_test:]
