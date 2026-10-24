from typing import Tuple

import numpy as np
from scipy.ndimage import filters
from skimage import morphology, transform

_SKEL_LEN_MASK = np.array([[0., 0., 0.], [0., 0., 1.], [np.sqrt(2.), 1., np.sqrt(2.)]])


def _process_img_morph(img, threshold=.5, scale=1):
    if scale > 1:
        up_img = transform.pyramid_expand(img, upscale=scale, order=3)
        img = (255. * up_img).astype(img.dtype)
    img_min, img_max = img.min(), img.max()
    bin_img = (img >= img_min + (img_max - img_min) * threshold)
    return img, bin_img


class ImageMorphology:
    """Representation of the morphological processing pipeline applied to an image.

    Attributes
    ----------
    image : (H, W) numpy.ndarray
        Input image.
    threshold : float
        Relative binarisation threshold.
    scale : int
        Upscaling factor.
    hires_image : (scale*H, scale*W) numpy.ndarray
        Upscaled version of `image`.
    binary_image : (scale*H, scale*W) numpy.ndarray
        Thresholded `hires_image`.
    skeleton : (scale*H, scale*W) numpy.ndarray
        Morphological skeleton of `binary_image`.
    distance_map : (scale*H, scale*W) numpy.ndarray
        Euclidean distance map from the boundaries in `binary_image`.
    """

    def __init__(self, image, threshold: float = .5, scale: int = 1):
        """
        Parameters
        ----------
        image : (H, W) array_like
            Input image.
        threshold : float, optional
            A relative threshold between 0 and 1. The upsampled image will be binarised at this fraction
            between its minimum and maximum values.
        scale : int, optional
            Upscaling factor for subpixel morphological analysis (>=1).
        """
        self.image = np.asarray(image)
        self.threshold = threshold
        self.scale = scale

        self._binary_image = None
        self._skeleton = None
        self._distance_map = None

        self.hires_image, self.binary_image = _process_img_morph(self.image, self.threshold, self.scale)

    @property
    def binary_image(self) -> np.ndarray:
        return self._binary_image
    
    @binary_image.setter
    def binary_image(self, value: np.ndarray) -> None:
        value = value.astype(self.hires_image.dtype)
        img_min, img_max = value.min(), value.max()
        bin_img = (value >= img_min + (img_max - img_min) * self.threshold)
        self._binary_image = bin_img
        self._skeleton = None
        self._distance_map = None
    
    @property
    def skeleton(self) -> np.ndarray:
        if self._skeleton is None:
            self._skeleton, self._distance_map = morphology.medial_axis(self.binary_image, return_distance=True)
        return self._skeleton
    
    @property
    def distance_map(self) -> np.ndarray:
        if self._distance_map is None:
            self._skeleton, self._distance_map = morphology.medial_axis(self.binary_image, return_distance=True)
        return self._distance_map

    @property
    def area(self) -> float:
        """Total area/image mass."""
        return self.binary_image.sum() / self.scale ** 2

    @property
    def stroke_length(self) -> float:
        """Length of the estimated skeleton."""
        skel = self.skeleton.astype(float)
        conv = filters.correlate(skel, _SKEL_LEN_MASK, mode='constant')
        up_length = np.einsum('ij,ij->', conv, skel)
        return up_length / self.scale

    @property
    def mean_thickness(self) -> float:
        """Mean thickness along the skeleton."""
        thickness = 2. * np.mean(self.distance_map[self.skeleton]) / self.scale
        return thickness

    @property
    def median_thickness(self) -> float:
        """Median thickness along the skeleton."""
        thickness = 2. * np.median(self.distance_map[self.skeleton]) / self.scale
        return thickness

    def downscale(self, image) -> np.ndarray:
        """Convenience method to map an image in the hi-res scale down to the original MNIST format.

        Parameters
        ----------
        image : (scale*H, scale*W) array_like
            High-resolution input image.

        Returns
        -------
        (H, W) numpy.ndarray
            Low-resolution `uint8` image.
        """
        image = np.asarray(image)
        if self.scale > 1:
            down_img = transform.pyramid_reduce(image, downscale=self.scale, order=3)
        else:
            down_img = image
        return (255. * down_img).astype(np.uint8)


class ImageMoments:
    """First- and second-order image moments.

    This class assumes that the vertical direction is indexed along the array's first axis,
    and horizontal along the second.

    Attributes
    ----------
    m00 : float
        Total mass.
    m10, m01 : float
        First-order moments (centroid).
    u20, u11, u02 : float
        Second-order central moments (covariance).
    """

    def __init__(self, img: np.ndarray):
        """
        Parameters
        ----------
        img : (H, W) array_like
            Input image whose moments to compute.
        """
        img = np.asarray(img, dtype=float)
        x = np.arange(img.shape[1])[None, :]
        y = np.arange(img.shape[0])[:, None]
        m00 = img.sum()
        m10 = (x * img).sum() / m00
        m01 = (y * img).sum() / m00
        m20 = (x ** 2 * img).sum() / m00
        m11 = (x * y * img).sum() / m00
        m02 = (y ** 2 * img).sum() / m00
        self.m00 = m00
        self.m10 = m10
        self.m01 = m01
        self.u20 = m20 - m10 ** 2
        self.u11 = m11 - m10 * m01
        self.u02 = m02 - m01 ** 2

    @property
    def centroid(self) -> Tuple[float, float]:
        """Image centroid."""
        return self.m10, self.m01

    @property
    def covariance(self) -> Tuple[float, float, float]:
        """Image's horizontal variance, cross-covariance and vertical variance."""
        return self.u20, self.u11, self.u02

    @property
    def axis_lengths(self) -> Tuple[float, float]:
        """Lenghts of the image's major and minor axes."""
        delta = .5 * np.hypot(2. * self.u11, self.u20 - self.u02)
        eig1 = .5 * (self.u20 + self.u02) + delta
        eig2 = .5 * (self.u20 + self.u02) - delta
        return np.sqrt(eig1), np.sqrt(eig2)

    @property
    def angle(self) -> float:
        """Orientation of the image's major axis."""
        return .5 * np.arctan2(2. * self.u11, self.u20 - self.u02)

    @property
    def horizontal_shear(self) -> float:
        """Image's horizontal shear."""
        return self.u11 / self.u02

    @property
    def vertical_shear(self) -> float:
        """Image's vertical shear."""
        return self.u11 / self.u20

