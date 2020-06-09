import torch
import numpy as np
import os
import glob
#import skvideo
#skvideo.setFFmpegPath("C:\\ffmpeg") # you need this before the import
import skvideo.io

def crop():
    current_path = os.path.dirname(__file__)
    resized_path = os.path.join(current_path, 'resized_data')
    dirs = glob.glob(os.path.join(current_path, 'data/*'))
    files = [ glob.glob(dir+'/*') for dir in dirs ]
    files = sum(files, []) # flatten

    ''' script for cropping '''
    for i, file in enumerate(files):
        os.system("ffmpeg -i %s -pix_fmt yuv420p -vf crop=96:96:42:24 %s.mp4" %
                 (file, os.path.join(resized_path, str(i))))

def preprocess(args):
    """
    Apply normalisation
    Transpose each video to (channel, nframe, img_size, img_size)
    """
    crop()
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, 'resized_data')
    vid_file = glob.glob(data_dir+'/*')
    #print(len(vid_file))
    videos = [skvideo.io.vread(vid) for vid in vid_file] # video size: (nframe, img_size, img_size, channel)
    # Normalising and appling transpose
    videos = [video.transpose(3, 0, 1, 2)/255.0 for video in videos ]
    return videos, curr_dir

def sample(video, T):
    #print(video.shape[0])
    start = np.random.randint(0, video.shape[1]-(T+1))
    end = start + T
    return video[:, start:end, :, :]

def randomVideo(videos, batch_size, T):
    x = []
    for i in range(batch_size):
        # Randomly Sample a video from the videos
        video = videos[ np.random.randint(1, len(videos)-1)]
        # Randomly sample the sequence of T frames from the video
        video = torch.Tensor(sample(video, T))
        x.append(video)
    x = torch.stack(x)
    return x

def save_video(fake_video, epoch, current_path):
    outputdata = fake_video * 255
    outputdata = outputdata.astype(np.uint8)
    dir_path = os.path.join(current_path, 'generated_videos')
    file_path = os.path.join(dir_path, 'fakeVideo_epoch-%d.mp4' % epoch)
    skvideo.io.vwrite(file_path, outputdata)



