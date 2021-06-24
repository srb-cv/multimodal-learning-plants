from typing import Callable, Sequence, Mapping,  Dict

import cv2
import numpy as np
from PIL import Image


class Compose:
    """
    Composes several transforms with arbitrary number of input and output parameters together.
    Input and output parameters of consecutive transforms must be compatible.
    """
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = transforms

    def __call__(self, *args):
        out = args
        for t in self.transforms:
            args = (out,) if not isinstance(out, tuple) else out
            out = t(*args)
        return out

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ModalityWiseTransform:
    """
    Combines separate uni-modal transforms (e.g. torchvision.transforms)
    into instance of DataTransform, where transforms applied modality-wise
    """
    def __init__(self, transforms: Mapping[str, Callable]):
        self.transforms = transforms

    def __call__(self, data: Dict) -> Dict:
        return {modality: transform(data[modality]) for modality, transform in self.transforms.items()}

    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.transforms)


class CropPlot:
    def __call__(self, image):
        return self.crop_plot_image(image)

    @staticmethod
    def crop_plot_image(image):
        image = np.array(image)
        image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        plot_contour = contours[0]
        rect = cv2.minAreaRect(plot_contour)
        box = np.int0(cv2.boxPoints(rect))
        width, height = map(int, rect[1])
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (width, height))

        return Image.fromarray(warped)
