import logging
import sys
import traceback
from typing import Dict, Tuple

import numpy as np
from PIL import Image as PillowImage

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
            if image.num_optical_paths > 1:
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
        return self._slide.total_pixel_matrix_dimensions[0]

    @property
    def level_dimensions(self) -> Tuple[Tuple[int, int], ...]:
        """Tuple[Tuple[int, int]]: Width and height of images at each level"""
        return self._slide.total_pixel_matrix_dimensions

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
        """Dict[str, PIL.Image.Image]: Mapping of DICOM SOP Instance UID to
        LABEL or OVERVIEW image

        """
        mapping = {}
        for image in self._slide.label_images + self._slide.overview_images:
            matrix = image.get_total_pixel_matrix()
            mapping[image.metadata.SOPInstanceUID] = self._convert_to_pil_image(
                matrix[:, :, :]
            )
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
            (Column, Row) offset of the region from the topleft hand pixel of
            of the total pixel matrix of the image at the base level 0
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
            pixel_indices=location,
            level=level,
            size=size,
            optical_path_index=1,
            focal_plane_index=1
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
        image_metadata = self._volume_images[0].metadata
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
            OPENSLIDE_BOUNDS_WIDTH: image_metadata.ImagedVolumeWidth,
            OPENSLIDE_BOUNDS_HEIGHT: image_metadata.ImagedVolumeHeight,
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

    def get_thumbnail(self, size: Tuple[int, int]) -> PillowImage:
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
        downsample = max(
            *(dim / thumb for dim, thumb in zip(self.dimensions, size))
        )
        level = self.get_best_level_for_downsample(downsample)
        tile = self.read_region((0, 0), level, self.level_dimensions[level])
        bg_color = '#{}'.format(
            self.properties.get(OPENSLIDE_BACKGROUND_COLOR, 'ffffff')
        )
        thumb = PillowImage.new('RGB', tile.size, bg_color)
        thumb.paste(tile, None, tile)
        thumb.thumbnail(size, PillowImage.ANTIALIAS)
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
