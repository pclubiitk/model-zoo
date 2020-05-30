import os
import glob

current_path = os.path.dirname(__file__)
resized_path = os.path.join(current_path, 'resized_data')
dirs = glob.glob(os.path.join(current_path, 'data/*'))
files = [ glob.glob(dir+'/*') for dir in dirs ]
files = sum(files, []) # flatten

''' script for cropping '''
for i, file in enumerate(files):
    os.system("ffmpeg -i %s -pix_fmt yuv420p -vf crop=96:96:42:24 %s.mp4" %
             (file, os.path.join(resized_path, str(i))))
