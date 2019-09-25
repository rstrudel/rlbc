import skvideo
import skvideo.io as skv
import numpy as np

from bc.settings import FFMPEG_PATH
skvideo.setFFmpegPath(FFMPEG_PATH)


def write_video(frames, path):
    skv.vwrite(path, np.array(frames).astype(np.uint8))
