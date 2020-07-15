import os
import json
import time
import numpy as np
from PIL import Image
from shutil import copyfile


def get_caption_text(filepath) :
    captions_dict = {}
    with open(filepath) as f:
        for line in f:
            line_split = line.split(sep='\t', maxsplit=1)
            caption = line_split[1][:-1]
            id_img = line_split[0].split(sep='#')[0]
            if id_img not in captions_dict:
                captions_dict[id_img] = [caption]
            else:
                captions_dict[id_img].append(caption)

    return captions_dict

def get_ids(filepath):
    ids = []
    with open(filepath) as f:
        for line in f:
            ids.append(line[:-1])

    return ids

def copy_files(dir_output, dir_images, ids):
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    for id in ids:
        path_input = os.path.join(dir_images, id)
        path_output = os.path.join(dir_output, id)
        copyfile(path_input, path_output)

def write_captions(dir_output, ids, captions_dict):
    output_path = os.path.join(dir_output, 'captions.txt')
    output = []
    for id in ids:
        cur_dict = {id : captions_dict[id]}
        output.append(json.dumps(cur_dict))

    with open(output_path, mode='w') as f:
        f.write('\n'.join(output))

def load_captions(captions_dir):
    file = os.path.join(captions_dir, 'captions.txt')
    captions_dict = {}
    with open(file) as f:
        for line in f:
            curr_dict = json.loads(line)
            for i,txt in curr_dict.items():
                captions_dict[i] = txt

    return captions_dict

def segregate(dir_images, token_filepath, captions_path):
    dir_output = {
                    'train':'train',
                    'dev' : 'dev',
                    'test' : 'test'
                    }
    captions_dict = get_caption_text(token_filepath)

    images = os.listdir(dir_images)

    ids_train = get_ids(captions_path['train'])
    ids_dev = get_ids(captions_path['dev'])
    ids_test = get_ids(captions_path['test'])

    copy_files(dir_output['train'], dir_images, ids_train)
    copy_files(dir_output['dev'], dir_images, ids_dev)
    copy_files(dir_output['test'], dir_images, ids_test)

    write_captions(dir_output['train'], ids_train, captions_dict)
    write_captions(dir_output['dev'], ids_dev, captions_dict)
    write_captions(dir_output['test'], ids_test, captions_dict)


if __name__ == '__main__' :
    dir_images = 'images'
    dir_text = 'text'
    filename_token = 'Flickr8k.token.txt'
    filename_train = 'Flickr_8k.trainImages.txt'
    filename_test = 'Flickr_8k.testImages.txt'
    filename_dev = 'Flickr_8k.devImages.txt'
    token_filepath = os.path.join(dir_text, filename_token)
    captions_path = {
                        'train': os.path.join(dir_text, filename_train),
                        'dev' : os.path.join(dir_text, filename_dev),
                        'test' : os.path.join(dir_text, filename_test)
                    }

    start = time.time()
    segregate(dir_images, token_filepath, captions_path)
    end = time.time()
    print('Preprocesssing of Flickr8k dataset done in :{0:.2f} mins'.format((end-start)/60))

## Preprocesssing of Flickr8k dataset done in :0.48 mins
