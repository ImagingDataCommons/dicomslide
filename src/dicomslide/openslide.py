import logging
import sys
import traceback
from typing import Dict, Tuple

import numpy as np
from PIL import Image as PillowImage

from dicomslide.enum import ImageFlavors
from dicomslide.slide import Slide

logger = logging.getLogger(__name__)

OPENSLIDE_COMMENT = 'openslide.comment'
OPENSLIDE_VENDOR = 'openslide.vendor'
OPENSLIDE_QUICKHASH1 = 'openslide.quickhash-1'
OPENSLIDE_BACKGROUND_COLOR = 'openslide.background-color'
OPENSLIDE_OBJECTIVE_POWER = 'openslide.objective-power'
OPENSLIDE_MPP_X = 'openslide.mpp-x'
OPENSLIDE_MPP_Y = 'openslide.mpp-y'
OPENSLIDE_BOUNDS_X = 'openslide.bounds-x'
OPENSLIDE_BOUNDS_Y = 'openslide.bounds-y'
OPENSLIDE_BOUNDS_WIDTH = 'openslide.bounds-width'
OPENSLIDE_BOUNDS_HEIGHT = 'openslide.bounds-height'


class OpenSlide:

    """Wrapper class that exposes data of a slide via the OpenSlide interface.

    Note
    ----
    There are two major differences between the OpenSlide interface exposed by
    this class and the interface exposed by :class:`dicomslide.Slide`::

        1. The OpenSlide API returns images as :class:`PIL.Image.Image` objects,
        while :class:`dicomslide.Slide` returns pixel arrays as
        :class:`numpy.ndarray` objects.
        2. The OpenSlide API specifies image dimensions and indices in
        column-major order (following the Pillow convention), while
        :class:`dicomslide.Slide` specifies array dimensions and indices in
        row-major order (following the NumPy convention).

    """

    def __init__(self, slide: Slide):
        """

        Parameters
        ----------
        slide: dicomslide.slide.Slide
            DICOM slide

        """
        self._slide = slide
        self._volume_images = self._slide.get_volume_images()
        for image in self._volume_images:
            if image.metadata.SamplesPerPixel != 3:
                raise ValueError(
                    'OpenSlide API only supports color images.'
                )
            if image.num_channels > 1:
                raise ValueError(
                    'OpenSlide API only supports images with a single '
                    'optical path.'
                )
            if image.num_focal_planes > 1:
                raise ValueError(
                    'OpenSlide API only supports images with a single '
                    'focal plane.'
                )

    @property
    def level_count(self) -> int:
        """int: Number of pyramid resolution levels"""
        return self._slide.num_levels

    @property
    def dimensions(self) -> Tuple[int, int]:
        """Tuple[int, int]: Width and height of images at base level 0"""
        return (
            self._slide.total_pixel_matrix_dimensions[0][1],
            self._slide.total_pixel_matrix_dimensions[0][0],
        )

    @property
    def level_dimensions(self) -> Tuple[Tuple[int, int], ...]:
        """Tuple[Tuple[int, int]]: Width and height of images at each level"""
        return tuple([
            (dimensions[1], dimensions[0])
            for dimensions in self._slide.total_pixel_matrix_dimensions
        ])

    @property
    def level_downsamples(self) -> Tuple[float, ...]:
        """Tuple[float]: Downsampling factor of images at each level with
        respect to the base level 0
        """
        return self._slide.downsampling_factors

    def _convert_to_pil_image(
        self,
        pixel_array: np.ndarray,
    ) -> PillowImage.Image:
        """Decode a frame and convert it into a Pillow Image object.

        Parameters
        ----------
        frame: bytes
            Frame item of the Pixel Data element of an image
        transfer_syntax_uid: str
            UID of the transfer syntax

        Returns
        -------
        PIL.Image.Image
            RGBA image

        """
        shape = pixel_array.shape
        image = PillowImage.frombuffer(
            data=pixel_array,
            size=(shape[1], shape[0]),
            mode='RGB'
        )
        return image.convert('RGBA')

    @property
    def associated_images(self) -> Dict[str, PillowImage.Image]:
        """Dict[str, PIL.Image.Image]: Mapping of image flavor (LABEL or
        OVERVIEW) to image

        """
        mapping = {}
        if len(self._slide.label_images) > 0:
            if len(self._slide.label_images) > 1:
                logger.warning(
                    'slide has more than one associated LABEL image'
                )
            image = self._slide.label_images[0]
            matrix = image.get_total_pixel_matrix()
            flavor = ImageFlavors.LABEL.value
            mapping[flavor] = self._convert_to_pil_image(matrix[:, :, :])
        if len(self._slide.overview_images) > 0:
            if len(self._slide.overview_images) > 1:
                logger.warning(
                    'slide has more than one associated OVERVIEW image'
                )
            image = self._slide.overview_images[0]
            matrix = image.get_total_pixel_matrix()
            flavor = ImageFlavors.OVERVIEW.value
            mapping[flavor] = self._convert_to_pil_image(matrix[:, :, :])
        return mapping

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int]
    ) -> PillowImage.Image:
        """Read region of a VOLUME (or THUMBNAIL) image at a given level.

        Parameters
        ----------
        location: Tuple[int, int]
            Zero-based (column, row) offset of the region from the topleft hand
            pixel of of the total pixel matrix of the image at the base level 0
        level: int
            Zero-based level index
        size: Tuple[int, int]
            Number of pixels columns and rows that should be read from the
            total pixel matrix at the specified level

        Returns
        -------
        PIL.Image.Image
            RGBA image

        """
        pixel_array = self._slide.get_image_region(
            offset=(location[1], location[0]),
            level=level,
            size=(size[1], size[0]),
            channel_index=0,
            focal_plane_index=0
        )
        return self._convert_to_pil_image(pixel_array)

    @property
    def properties(self) -> Dict[str, str]:
        """Metadata about the slide.

        Returns
        -------
        Dict[str, str]
            OpenSlide properties

        """
        image = self._volume_images[0]
        image_metadata = image.metadata
        pixel_spacing = (
            image_metadata
            .SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing
        )
        # TODO: Does ImageOrientationSlide need to be taken into account?
        x_offset = (
            image_metadata
            .TotalPixelMatrixOriginSequence[0]
            .XOffsetInSlideCoordinateSystem
        )
        y_offset = (
            image_metadata
            .TotalPixelMatrixOriginSequence[0]
            .YOffsetInSlideCoordinateSystem
        )

        return {
            OPENSLIDE_COMMENT: 'DICOM',
            OPENSLIDE_VENDOR: image_metadata.Manufacturer,
            OPENSLIDE_QUICKHASH1: str(hash(self._slide)),
            # TODO: Consider using RecommendedAbsentPixelCIELabValue if
            # available. However, that would need to be used during decoding
            # when the ICC Profile is applied.
            OPENSLIDE_BACKGROUND_COLOR: 'FFFFFF',
            OPENSLIDE_OBJECTIVE_POWER: getattr(
                image_metadata.OpticalPathSequence[0],
                'ObjectiveLensPower',
                ''
            ),
            OPENSLIDE_MPP_X: str(pixel_spacing[1] * 10**3),
            OPENSLIDE_MPP_Y: str(pixel_spacing[1] * 10**3),
            OPENSLIDE_BOUNDS_X: x_offset,
            OPENSLIDE_BOUNDS_Y: y_offset,
            OPENSLIDE_BOUNDS_WIDTH: str(image.physical_size[1]),
            OPENSLIDE_BOUNDS_HEIGHT: str(image.physical_size[0]),
        }

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """Compute best level for displaying the given downsample.

        Parameters
        ----------
        downsample: float
            Desired downsample factor

        Returns
        -------
        int
            Zero-based level index

        """
        level_downsamples = np.array(self.level_downsamples)
        distances = np.abs(level_downsamples - downsample)
        return int(np.argmin(distances))

    def get_thumbnail(self, size: Tuple[int, int]) -> PillowImage.Image:
        """Create a thumbnail of the slide.

        Parameters
        ----------
        size: Tuple[int, int]
            Number of pixels columns and rows that the thumbnail should have

        Returns
        -------
        PIL.Image.Image
            RGB image

        """
        downsampling_factor = max(
            *(dim / thumb for dim, thumb in zip(self.dimensions, size))
        )
        level = self.get_best_level_for_downsample(downsampling_factor)
        tile = self.read_region((0, 0), level, self.level_dimensions[level])
        background_color = '#{}'.format(
            self.properties.get(OPENSLIDE_BACKGROUND_COLOR, 'ffffff')
        )
        thumb = PillowImage.new('RGB', tile.size, background_color)
        thumb.paste(tile, None, tile)
        try:
            resamping_method = PillowImage.Resampling.LANCZOS  # type: ignore
        except AttributeError:
            # May be using a version of Pillow before 10.0.0, especially if
            # using an older version of Python
            resamping_method = PillowImage.ANTIALIAS

        thumb.thumbnail(size, resamping_method)
        return thumb

    def __enter__(self):
        return self

    def __exit__(self, except_type, except_value, except_trace) -> None:
        self.close()
        if except_value:
            sys.stderr.write('Error while reading slide:\n{except_value}')
            for tb in traceback.format_tb(except_trace):
                sys.stderr.write(tb)
            raise

    def close(self) -> None:
        # Method is only implemented for compability with the OpenSlide API.
        pass
