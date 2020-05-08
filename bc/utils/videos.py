import os
import skvideo
import skvideo.io as skv
import numpy as np

if 'FFMPEG_PATH' in os.environ:
    skvideo.setFFmpegPath(os.environ['FFMPEG_PATH'])


def write_video(frames, path):
    skv.vwrite(path, np.array(frames).astype(np.uint8))
