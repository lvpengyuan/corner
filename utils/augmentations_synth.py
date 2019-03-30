import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from shapely.geometry import box, Polygon
import math



class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        if boxes.size > 0:
            boxes[:, 0] *= width
            boxes[:, 1] *= height
            boxes[:, 2] *= width
            boxes[:, 3] *= height
            boxes[:, 4] *= width
            boxes[:, 5] *= height
            boxes[:, 6] *= width
            boxes[:, 7] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        if boxes.size > 0:
            boxes[:, 0] /= width
            boxes[:, 1] /= height
            boxes[:, 2] /= width
            boxes[:, 3] /= height
            boxes[:, 4] /= width
            boxes[:, 5] /= height
            boxes[:, 6] /= width
            boxes[:, 7] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=(300, 512)):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, self.size)
        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels




class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        if boxes.size > 0:
            boxes[:, :2] += (int(left), int(top))
            boxes[:, 2:4] += (int(left), int(top))
            boxes[:, 4:6] += (int(left), int(top))
            boxes[:, 6:8] += (int(left), int(top))

        return image, boxes, labels


class RandomRotate(object):
    def __call__(self, image, boxes, labels):
        height, width, _ = image.shape
        n_boxes = boxes.copy()
        rand_num = random.rand()
        if  rand_num< 0.5:
            pass
        elif rand_num >= 0.5  and rand_num < 0.75:
            ## flip 90
            M = cv2.getRotationMatrix2D((height/2,width/2),-90,1)
            image = cv2.warpAffine(image,M,(height,width))
            if n_boxes.size > 0:
                n_boxes[:, 0] = height - boxes[:, 7]
                n_boxes[:, 1] = boxes[:, 6]
                n_boxes[:, 2] = height - boxes[:, 1]
                n_boxes[:, 3] = boxes[:, 0]
                n_boxes[:, 4] = height - boxes[:, 3]
                n_boxes[:, 5] = boxes[:, 2]
                n_boxes[:, 6] = height - boxes[:, 5]
                n_boxes[:, 7] = boxes[:, 4]
        else:
            # flip -90
            M = cv2.getRotationMatrix2D((height/2,width/2),90,1)
            image = cv2.warpAffine(image,M,(height,width))
            if n_boxes.size > 0:
                n_boxes[:, 0] = boxes[:, 3]
                n_boxes[:, 1] = width - boxes[:, 2]
                n_boxes[:, 2] = boxes[:, 5]
                n_boxes[:, 3] = width - boxes[:, 4]
                n_boxes[:, 4] = boxes[:, 7]
                n_boxes[:, 5] = width - boxes[:, 6]
                n_boxes[:, 6] = boxes[:, 1]
                n_boxes[:, 7] = width - boxes[:, 0]
        return image, n_boxes, labels



class RandomMirror(object):
    def __call__(self, image, boxes, labels):
        _, width, _ = image.shape
        n_boxes = boxes.copy()
        if random.randint(2):
            image = image[:, ::-1]
            # boxes[:, 0::2] = width - boxes[:, 2::-2]
            if n_boxes.size > 0:
                n_boxes[:, 0] = width - boxes[:, 2]
                n_boxes[:, 1] = boxes[:, 3]
                n_boxes[:, 2] = width - boxes[:, 0]
                n_boxes[:, 3] = boxes[:, 1]
                n_boxes[:, 4] = width - boxes[:, 6]
                n_boxes[:, 5] = boxes[:, 7]
                n_boxes[:, 6] = width - boxes[:, 4]
                n_boxes[:, 7] = boxes[:, 5]

        return image, n_boxes, labels




class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


class SynthSSDAugmentation(object):
    def __init__(self, size=(512, 512), mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels=None):
        return self.augment(img, boxes)