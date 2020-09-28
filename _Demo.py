import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from demo import load_checkpoints
from demo import make_animation
from skimage import img_as_ubyte


import warnings
warnings.filterwarnings("ignore")

source_image = imageio.imread('material/02.png')
driving_video = imageio.mimread('material/04.mp4')

#Resize image and video to 256x256
source_image = resize(source_image, (256, 256))[..., :3]
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]


def display(source, driving, generated=None):
    fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))

    ims = []
    for i in range(len(driving)):
        cols = [source]
        cols.append(driving[i])
        if generated is not None:
            cols.append(generated[i])
        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    plt.close()
    return ani

generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml',
                            checkpoint_path='material/vox-cpk.pth.tar')

predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

#save resulting video
imageio.mimsave('generated/generated.mp4', [img_as_ubyte(frame) for frame in predictions])
#video can be downloaded from /content folder


