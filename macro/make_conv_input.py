import TOOLS
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    start = 0
    stop = 50000
    variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_03.npy"
    image_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_03_image_0.05M_10_curvT.npy"

    data = np.load(variable_name)
    print("File load successfully")

    tr1s, tr2s = data[start:stop, 3:9], data[start:stop, 25:31]
    #t = np.arange(-0.005, 0.005, 0.00005)
    t = np.arange(-10, 10, 0.1)

    images = []
    for tr1, tr2 in tqdm(zip(tr1s, tr2s)):
        track1, track2 = TOOLS.t_tracker(tr1, tr2, t, curvature=True)
        image = []
        for _track1 in track1:
            img = []
            for _track2 in track2:
                img.append(_track1 - _track2)
            image.append(img)
        image = np.reshape(image, [200, 200, 3])
        images.append(image)

    images = np.array(images, dtype=float)
    print(images.shape)

    np.save(image_name, images, fix_imports=True)
