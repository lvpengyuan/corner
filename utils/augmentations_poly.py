import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from shapely.geometry import box, Polygon
import math

def get_tight_rect(points):
    tpoints = []
    for i in range(len(points)):
        tpoints.append([int(points[i][0]), int(points[i][1])])
    bounding_box = cv2.minAreaRect(np.array(tpoints))
    points = cv2.boxPoints(bounding_box)
    points = list(points)
    ps = sorted(points,key = lambda x:x[0])

    if ps[1][1] > ps[0][1]:
        px1 = ps[0][0]
        py1 = ps[0][1]
        px4 = ps[1][0]
        py4 = ps[1][1]
    else:
        px1 = ps[1][0]
        py1 = ps[1][1]
        px4 = ps[0][0]
        py4 = ps[0][1]
    if ps[3][1] > ps[2][1]:
        px2 = ps[2][0]
        py2 = ps[2][1]
        px3 = ps[3][0]
        py3 = ps[3][1]
    else:
        px2 = ps[3][0]
        py2 = ps[3][1]
        px3 = ps[2][0]
        py3 = ps[2][1]

    return [px1, py1, px2, py2, px3, py3, px4, py4]



def jaccard_numpy(box_a, box_b):
    p_b = Polygon([[box_b[0], box_b[1]], [box_b[2], box_b[3]], [box_b[4], box_b[5]], [box_b[6], box_b[7]]])
    ious = []
    re_rect = []
    for b_a in box_a:
        p_a = Polygon([[b_a[0], b_a[1]], [b_a[2], b_a[3]], [b_a[4], b_a[5]], [b_a[6], b_a[7]]])
        intersect = p_b.intersection(p_a)
        iou = intersect.area*1.0/(p_a.area + p_b.area - intersect.area)
        if iou > 0 and intersect.area > 30:
            ious.append(iou)
            pts = intersect.boundary.coords[:-1]
            mb = get_tight_rect(pts)
            re_rect.append(mb)
        else:
            ious.append(0)
            re_rect.append([])

    return ious, re_rect



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
    def __init__(self, size=(512, 512)):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size[1], self.size[0]))
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


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])
                rect_2 = np.array([int(left), int(top), int(left+w), int(top), int(left+w), int(top+h), int(left), int(top+h)])
                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                if boxes.size == 0:
                    return current_image, boxes, None


                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap, crop_boxes = jaccard_numpy(boxes, rect_2)
                # is min and max overlap constraint satisfied? if not try again
                # if overlap.min() < min_iou and max_iou < overlap.max():

                if min(overlap) < min_iou :
                    continue

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:4] + boxes[:, 4:6] + boxes[:, 6:]) / 4.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue
                current_boxes = []
                current_labels = None

                for i in range(len(overlap)):
                    if overlap[i] > 0:
                        c_b = crop_boxes[i]
                        if c_b[0] < rect[0]:
                            c_b[0] = rect[0]
                        if c_b[1] < rect[1]:
                            c_b[1] = rect[1]
                        if c_b[2] > rect[2]:
                            c_b[2] = rect[2]
                        if c_b[3] < rect[1]:
                            c_b[3] = rect[1]
                        if c_b[4] > rect[2]:
                            c_b[4] = rect[2]
                        if c_b[5] > rect[3]:
                            c_b[5] = rect[3]
                        if c_b[6] < rect[0]:
                            c_b[6] = rect[0]
                        if c_b[7] > rect[3]:
                            c_b[7] = rect[3]

                        if mask[i] > 0:
                            current_boxes.append(c_b)

                if len(current_boxes) == 0:
                    continue

                current_boxes = np.array(current_boxes)
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:4] -= rect[:2]
                current_boxes[:, 4:6] -= rect[:2]
                current_boxes[:, 6:] -= rect[:2]

                return current_image, current_boxes, current_labels


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
        if  rand_num< 0.8:
            pass
        elif rand_num >= 0.8  and rand_num < 0.9:
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


class SSDAugmentation(object):
    def __init__(self, size=(512, 512), mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            # ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            
            ToPercentCoords(),
            Resize(self.size),

            # ToAbsoluteCoords(),
            # RandomRotate(),
            # ToPercentCoords(),
            
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels=None):
        return self.augment(img, boxes)
