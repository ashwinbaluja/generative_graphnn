import numpy as np
import cv2
import numpy as np
import tensorflow as tf
import random
import tensorflow_datasets as tfds

ds = tfds.load('imagenet_v2', split='test', shuffle_files=True, as_supervised=True)
#assert isinstance(ds, tf.data.Dataset)
#
ds2 = ds.take(1)
size = 64
circlemin = 1
circlemax = 6
numcircles = 6

def crop(im, size):
    sizes = im.shape[:2]
    ratio = size / min(sizes)
    resized = cv2.resize(im, (int(sizes[0] * ratio), int(sizes[1] * ratio)))
    cropped = resized[:size, :size, :]
    zeros = np.zeros((size, size, 3), dtype=np.uint8)
    zeros[:cropped.shape[0], :cropped.shape[1], :] = cropped
    return zeros

def image_process(im, imsize, circlemin, circlemax, numcircles):
    cropped = crop(im, imsize)
    circled = cropped
    for i in range(numcircles):
        circled = cv2.circle(
            circled, (random.randint(0, imsize-1), random.randint(0, imsize-1)), random.randint(circlemin, circlemax), (255, 255, 255), -1
        )
    
    return circled

numpyds = tfds.as_numpy(ds)

transformedX = []
transformedY = []
counter = 0
for i in numpyds:
    counter += 1
    if counter % 100 == 0:
        print(counter)
    
    transformedY.append(crop(i[0], size) / 255)
    transformedX.append(image_process(i[0], size, circlemin, circlemax, numcircles) / 255)
    cv2.imshow("frame", image_process(i[0], size, circlemin, circlemax, numcircles))
    cv2.waitKey(0)

x = np.array(transformedX, dtype=np.float32)
y = np.array(transformedY, dtype=np.float32)

np.save("x.npy", x)
np.save("y.npy", y)

print(x.shape)
print(y.shape)