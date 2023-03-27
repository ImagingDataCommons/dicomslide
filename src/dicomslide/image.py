import itertools
import logging
from hashlib import sha256
from typing import List, Optional, Tuple

import highdicom as hd
import numpy as np
from dicomweb_client import DICOMClient
from pydicom.dataset import Dataset
from pydicom.uid import (
    ParametricMapStorage,
    SegmentationStorage,
    VLWholeSlideMicroscopyImageStorage,
)
from scipy.ndimage import rotate

from dicomslide._channel import _get_channel_info
from dicomslide.enum import ChannelTypes
from dicomslide.matrix import TotalPixelMatrix
from dicomslide.tile import compute_frame_positions
from dicomslide.utils import _encode_dataset

logger = logging.getLogger(__name__)


class TiledImage:

    """A tiled DICOM image.

    An instance of the class represents a tiled DICOM image instance and
    provides methods for convenient and efficient access of image metadata and
    pixel data from a DICOMweb server (or another source for which the
    :class:`dicomweb_client.DICOMClient` protocol has been implemented).

    A tiled image is hereby defined as a DICOM image instance that contains
    the Total Pixel Matrix Rows and Total Pixel Matrix Columns attributes.

    The class is designed to be independent of a particular DICOM Information
    Object Definition (IOD) or SOP Class and support various different types of
    DICOM images, including VL Whole Slide Microscopy Image, Segmentation, and
    Parametric Map.

    Each image is associated with one or more
    :class:`dicomslide.TotalPixelMatrix` instances, one for each unique
    combination of channel and focal plane.
    The definition of a channel is specific to a particular IOD. For example,
    in case of VL Whole Slide Microscopy Image, a channel corresponds to an
    optical path, whereas in case of a Segmentation, a channel corresponds to a
    segment.

    Examples
    --------
    >>> image = TiledImage(...)
    >>> print(image.metadata)  # pydicom.Dataset
    >>> print(image.metadata.BitsAllocated)
    >>> print(image.metadata.TotalPixelMatrixRows)
    >>> pixel_matrix = image.get_tota_pixel_matrix(channel_index=0)
    >>> print(pixel_matrix.dtype)
    >>> print(pixel_matrix.shape)
    >>> print(pixel_matrix[:1000, 350:750, :])  # numpy.ndarray

    """

    def __init__(
        self,
        client: DICOMClient,
        image_metadata: Dataset,
        max_frame_cache_size: int = 6
    ) -> None:
        """Construct object.

        Parameters
        ----------
        client: dicomweb_client.api.DICOMClient
            DICOMweb client
        image_metadata: pydicom.dataset.Dataset
            Metadata of a tiled DICOM image
        max_frame_cache_size: int, optional
            Maximum number of frames that should be cached to avoid repeated
            retrieval requests

        Note
        ----
        If `image_metadata` is the metadata of a color image, it should contain
        the ICC Profile element to enable color management. The value of this
        element may be considered bulkdata and therefore may have to be
        retrieved separately over DICOMweb.

        """
        if not isinstance(client, DICOMClient):
            raise TypeError(
                'Argument "client" must have type dicomweb_client.DICOMClient.'
            )
        self._client = client
        if not isinstance(image_metadata, Dataset):
            raise TypeError(
                'Argument "image_metadata" must have type pydicom.Dataset.'
            )
        self._metadata = image_metadata
        # The Total Pixel Matrix Focal Planes attribute must be present in case
        # of Dimension Organization TILED_FULL and may be present otherwise.
        # If it is missing, we need to determine the position of focal planes by
        # either looking up the values in the items in the Per-Frame Functional
        # Groups Sequence attribute or by computing the values if the attribute
        # is not included in the dataset.
        try:
            self._number_of_focal_planes = getattr(
                self._metadata,
                'TotalPixelMatrixFocalPlanes'
            )
            self._frame_positions = None
        except AttributeError:
            self._frame_positions = compute_frame_positions(self._metadata)
            focal_plane_indices = self._frame_positions[3]
            self._number_of_focal_planes = len(np.unique(focal_plane_indices))

        channels, _ = _get_channel_info(self._metadata)
        self._channel_identifiers = tuple([
            ch.channel_identifier for ch in channels
        ])
        self._number_of_channels = len(self._channel_identifiers)

        self._pixel_spacing = (
            float(
                self
                .metadata
                .SharedFunctionalGroupsSequence[0]
                .PixelMeasuresSequence[0]
                .PixelSpacing[0]
            ),
            float(
                self
                .metadata
                .SharedFunctionalGroupsSequence[0]
                .PixelMeasuresSequence[0]
                .PixelSpacing[1]
            ),
        )

        iterator = itertools.product(
            range(self._number_of_channels),
            range(self._number_of_focal_planes),
        )
        self._total_pixel_matrix_lut = {
            (channel_index, focal_plane_index): TotalPixelMatrix(
                client=self._client,
                image_metadata=self._metadata,
                channel_index=channel_index,
                focal_plane_index=focal_plane_index,
                max_frame_cache_size=max_frame_cache_size
            )
            for channel_index, focal_plane_index in iterator
        }

        encoded_dataset = _encode_dataset(self._metadata)
        self._quickhash = sha256(encoded_dataset).hexdigest()

        image_orientation = self.metadata.ImageOrientationSlide
        image_origin = self.metadata.TotalPixelMatrixOriginSequence[0]
        image_position = (
            float(image_origin.XOffsetInSlideCoordinateSystem),
            float(image_origin.YOffsetInSlideCoordinateSystem),
            0,  # assume that focal planes are parallel to slide surface
        )
        self._ref2pix_transformer = hd.spatial.ReferenceToPixelTransformer(
            image_orientation=image_orientation,
            image_position=image_position,
            pixel_spacing=self._pixel_spacing,
        )
        self._pix2ref_transformer = hd.spatial.PixelToReferenceTransformer(
            image_orientation=image_orientation,
            image_position=image_position,
            pixel_spacing=self._pixel_spacing,
        )

    def __hash__(self) -> int:
        return hash(self._quickhash)

    @property
    def metadata(self) -> Dataset:
        """pydicom.dataset.Dataset: Image metadata"""
        return self._metadata

    @property
    def num_channels(self) -> int:
        """int: Number of channels"""
        return self._number_of_channels

    def map_pixel_indices_to_slide_coordinates(
        self,
        pixel_indices: np.ndarray,
    ) -> np.ndarray:
        """Map pixel indices to slide coordinates.

        Parameters
        ----------
        pixel_indices: numpy.ndarray
            Zero-based (row, column) indices into the total pixel matrix of the
            image

        Returns
        -------
        numpy.ndarray
            Zero-based (x, y, z) coordinates in the slide coordinate system in
            millimeter

        """
        coordinates = np.stack(
            [
                pixel_indices[:, 1],
                pixel_indices[:, 0],
            ],
            axis=1
        )
        return self._pix2ref_transformer(coordinates)

    def get_slide_offset(
        self,
        pixel_indices: Tuple[int, int],
    ) -> Tuple[float, float]:
        """Get slide coordinates for a given total pixel matrix position.

        Parameters
        ----------
        pixel_indices: Tuple[int, int]
            Zero-based (row, column) offset in the total pixel matrix

        Returns
        -------
        Tuple[float, float]
            Zero-based (x, y) position on the slide in the slide coordinate
            system in millimeter

        """
        row_index, column_index = pixel_indices
        slide_coordinates = self._pix2ref_transformer(
            np.array([[column_index, row_index]])
        )
        return (slide_coordinates[0][0], slide_coordinates[0][1])

    def map_slide_coordinates_to_pixel_indices(
        self,
        slide_coordinates: np.ndarray,
    ) -> np.ndarray:
        """Map slide coordinates to pixel indices.

        Parameters
        ----------
        slide_coordinates: numpy.ndarray
            Zero-based (x, y, z) coordinates in the slide coordinate system in
            millimeter

        Returns
        -------
        numpy.ndarray
            Zero-based (row, column) indices into the total pixel matrix of the
            image

        """
        pixel_coordinates = self._ref2pix_transformer(slide_coordinates)
        return np.stack(
            [
                pixel_coordinates[:, 1],
                pixel_coordinates[:, 0],
            ],
            axis=1
        )

    def get_pixel_indices(
        self,
        offset: Tuple[float, float],
    ) -> Tuple[int, int]:
        """Get indices into total pixel matrix for a given slide position.

        Parameters
        ----------
        offset: Tuple[float, float]
            Zero-based (x, y) offset in the slide coordinate system in
            millimeter

        Returns
        -------
        Tuple[int, int]
            Zero-based (row, column) position in the total pixel matrix

        Note
        ----
        Pixel position may be negativ or extend beyond the size of the total
        pixel matrix if slide position at `offset` does fall into a region on
        the slide that was not imaged.

        """
        slide_coordinates = np.array([[offset[0], offset[1], 0.0]])
        pixel_indices = self._ref2pix_transformer(slide_coordinates)
        return (
            int(pixel_indices[0, 1]),
            int(pixel_indices[0, 0]),
        )

    def get_rotation(self) -> float:
        """Get angle to rotate image such that it aligns with slide.

        We want to align the image with the slide coordinate system such that
        the axes of the total pixel matrix are aligned with the X and Y axes
        of the slide coordinate system to ensure that spatial coordinates of
        graphic region of interest (ROI) annotations are aligned with the
        source image region.

        Returns
        -------
        float
            Counterclockwise angle of rotation

        """
        image_orientation = self.metadata.ImageOrientationSlide
        radians = np.arctan2(-image_orientation[3], image_orientation[0])
        degrees = radians * 180.0 / np.pi
        degrees -= 180.0

        # Images are expected to be rotated in plane parallel to the slide
        # surface and the rows and columns of the image are expected to be
        # parallel to the axes of the slide.
        if degrees not in (-0.0, -90.0, -180.0, -270.0):
            logger.warning(
                'encountered unexpected image orientation: '
                f'{image_orientation}: {degrees}'
            )

        return degrees

    @property
    def frame_of_reference_uid(self) -> str:
        """str: Unique identifier of the frame of reference"""
        return str(self._metadata.FrameOfReferenceUID)

    @property
    def physical_offset(self) -> Tuple[float, float]:
        """Tuple[float, float]: Offset of the total pixel matrix from the
        origin of the frame of reference along the X and Y axes of the slide
        coordinate system in millimeter

        """
        image_origin = self._metadata.TotalPixelMatrixOriginSequence[0]
        return (
            float(image_origin.XOffsetInSlideCoordinateSystem),
            float(image_origin.YOffsetInSlideCoordinateSystem),
        )

    @property
    def physical_size(self) -> Tuple[float, float]:
        """Tuple[float, float]: Size of the total pixel matrix along the X and
        Y axes of the slide coordinate system in millimeter

        """
        x_endpoint, y_endpoint = self.get_slide_offset(pixel_indices=self.size)
        x_offset, y_offset = self.physical_offset
        return (
            abs(x_endpoint - x_offset),
            abs(y_endpoint - y_offset),
        )

    @property
    def size(self) -> Tuple[int, int]:
        """Tuple[int, int]: Number of total pixel matrix rows and columns"""
        return (
            int(self._metadata.TotalPixelMatrixRows),
            int(self._metadata.TotalPixelMatrixColumns),
        )

    def get_references(
        self,
        sop_class_uid: Optional[str] = None
    ) -> List[Tuple[str, str, str]]:
        """Get unique identifiers of referenced instances.

        Parameters
        ----------
        sop_class_uid: str
            SOP Class UID of instances for which references should be obtained

        Returns
        -------
        List[Tuple[str, str, str]]
            Study, Series, and SOP Instance UID of each referenced image

        """
        uids: List[Tuple[str, str, str]] = []
        study_instance_uid = self._metadata.StudyInstanceUID
        for series in getattr(self._metadata, 'ReferencedSeriesSequence', []):
            series_instance_uid = series.SeriesInstanceUID
            for instance in series.ReferencedInstanceSequence:
                sop_instance_uid = instance.ReferencedSOPInstanceUID
                if sop_class_uid is not None:
                    if sop_class_uid != instance.ReferencedSOPClassUID:
                        continue
                uids.append(
                    (
                        study_instance_uid,
                        series_instance_uid,
                        sop_instance_uid,
                    )
                )
        return uids

    def get_channel_index(self, channel_identifier: str) -> int:
        """Get index of a channel.

        The nature of the channel is specific to the SOP Class for the image.
        For example, in case of DICOM VL Whole Slide Microscopy Image, a
        channel is an optical path and in case of a DICOM Segmentation, a
        channel is a segment.

        Parameters
        ----------
        channel_identifier: str
            Identifier of a channel

        Returns
        -------
        int
            Zero-based index into channels along the direction defined by
            successive items of the corresponding attribute.

        Raises
        ------
        ValueError
            When no channel is found for `channel_identifier`

        """
        try:
            index = self._channel_identifiers.index(channel_identifier)
        except ValueError:
            raise ValueError(
                f'Image "{self._metadata.SOPInstanceUID}" does not have an '
                f'optical path with identifier "{channel_identifier}".'
            )
        return index

    @property
    def channel_type(self) -> ChannelTypes:
        """dicomslide.ChannelTypes: type of channels"""
        sop_class_uid = self._metadata.SOPClassUID
        sop_instance_uid = self._metadata.SOPInstanceUID
        lut = {
            VLWholeSlideMicroscopyImageStorage: ChannelTypes.OPTICAL_PATH,
            SegmentationStorage: ChannelTypes.SEGMENT,
            ParametricMapStorage: ChannelTypes.PARAMETER,
        }
        try:
            return lut[sop_class_uid]
        except IndexError:
            raise ValueError(
                'Could not determine channel type for '
                f'image "{sop_instance_uid}" of SOP Class "{sop_class_uid}".'
            )

    def get_channel_identifier(self, channel_index: int) -> str:
        """Get identifier of a channel.

        Parameters
        ----------
        channel_index: int
            Zero-based index into channels along the direction defined by
            successive items of the appropriate DICOM attribute.

        Returns
        -------
        str
            Channel identifier

        Raises
        ------
        ValueError
            When no channel is found for `channel_index`

        """
        try:
            return self._channel_identifiers[channel_index]
        except IndexError:
            raise ValueError(
                f'Image "{self._metadata.SOPInstanceUID}" does not have a '
                f'channel {channel_index}.'
            )

    @property
    def num_focal_planes(self) -> int:
        """int: Number of focal planes"""
        return self._number_of_focal_planes

    def get_focal_plane_index(self, focal_plane_offset: float) -> int:
        """Get index of a focal plane.

        Parameters
        ----------
        focal_plane_offset: float
            Offset of the focal plane from the from the slide surface along the
            Z axis of the slide coordinate system in micrometers

        Returns
        -------
        int
            Zero-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute.
            Values must be in the range [1, Total Pixel Matrix Focal Planes]

        Raises
        ------
        ValueError
            When no focal plane is found for `focal_plane_offset`

        """
        # Computing the focal plane index may be expensive, so only do it if
        # there are multiple focal planes.
        if self.num_focal_planes == 1:
            return 0
        if self._frame_positions is None:
            self._frame_positions = compute_frame_positions(self._metadata)
        slide_positions = self._frame_positions[1]
        focal_plane_indices = self._frame_positions[3]
        index = slide_positions[:, 2] == focal_plane_offset
        if np.sum(index) == 0:
            raise ValueError(
                f'Image "{self._metadata.SOPInstanceUID}" does not have a '
                f'focal plane with z offset {focal_plane_offset}.'
            )
        return focal_plane_indices[index][0]

    def get_focal_plane_offset(self, focal_plane_index: int) -> float:
        """Get z offset in slide coordinate system of a focal plane.

        Parameters
        ----------
        focal_plane_index: int
            Zero-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute.
            Values must be in the range [0, Total Pixel Matrix Focal Planes).

        Returns
        -------
        float
            Offset of the focal plane from the from the slide surface along the
            Z axis of the slide coordinate system in micrometers

        Raises
        ------
        ValueError
            When no focal plane is found for `focal_plane_index`

        """
        # TODO: cache focal plane offsets
        origin_item = self._metadata.TotalPixelMatrixOriginSequence[0]
        dim_org_type = hd.DimensionOrganizationTypeValues(
            getattr(
                self._metadata,
                'DimensionOrganizationType',
                hd.DimensionOrganizationTypeValues.TILED_SPARSE.value
            )
        )
        pixel_measures_item = (
            self._metadata
            .SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
        )
        if dim_org_type == hd.DimensionOrganizationTypeValues.TILED_FULL:
            # In case of Dimension Organization Type TILED_FULL, the absolute
            # position of focal planes is not explicitly specified and the
            # plane is assumed to be located at the slide surface.
            z_offset = float(
                getattr(
                    origin_item,
                    'ZOffsetInSlideCoordinateSytem',
                    '0.0'
                )
            )
            spacing_between_slices = float(
                getattr(pixel_measures_item, 'SpacingBetweenSlices', 1.0)
            )
            num_slices = self._metadata.TotalPixelMatrixFocalPlanes
            if focal_plane_index > num_slices:
                raise ValueError(
                    f'Image "{self._metadata.SOPInstanceUID}" does not have a '
                    f'focal plane with index {focal_plane_index}.'
                )
            if num_slices == 1:
                return z_offset
            else:
                # Note that Spacing Between Slices has millimeter unit while
                # the Z Offset in Slide Coordinate System has micrometer unit.
                return z_offset + (spacing_between_slices * num_slices * 10**3)
        else:
            if self._frame_positions is None:
                self._frame_positions = compute_frame_positions(self._metadata)
            slide_positions = self._frame_positions[1]
            focal_plane_indices = self._frame_positions[3]
            index = focal_plane_indices == focal_plane_index
            if np.sum(index) == 0:
                raise ValueError(
                    f'Image "{self._metadata.SOPInstanceUID}" does not have a '
                    f'focal plane with index {focal_plane_index}.'
                )
            return slide_positions[index, 2][0]

    def get_total_pixel_matrix(
        self,
        channel_index: int = 0,
        focal_plane_index: int = 0
    ) -> TotalPixelMatrix:
        """Get total pixel matrix for a given optical path and focal plane.

        Parameters
        ----------
        channel_index: int, optional
            Zero-based index into channels along the direction defined by
            successive items of the appropriate DICOM attribute.
        focal_plane_index: int, optional
            Zero-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute.
            Values must be in the range [0, Total Pixel Matrix Focal Planes).

        Returns
        -------
        dicomslide.TotalPixelMatrix
            Total Pixel Matrix

        """
        if (
            channel_index < 0 or
            channel_index >= self.num_channels
        ):
            raise ValueError(
                'Argument "channel_index" must be a zero-based index '
                f'in range [0, {self.num_channels}).'
            )
        if (
            focal_plane_index < 0 or
            focal_plane_index >= self.num_focal_planes
        ):
            raise ValueError(
                'Argument "focal_plane_index" must be a zero-based index '
                f'in range [0, {self.num_focal_planes}).'
            )

        key = (channel_index, focal_plane_index)
        try:
            return self._total_pixel_matrix_lut[key]
        except KeyError:
            raise ValueError(
                'Could not find a total pixel matrix for '
                f'optical path {channel_index} and '
                f'focal plane {focal_plane_index}.'
            )

    def get_image_region(
        self,
        offset: Tuple[int, int],
        size: Tuple[int, int],
        channel_index: int = 0,
        focal_plane_index: int = 0
    ) -> np.ndarray:
        """Get image region.

        Parameters
        ----------
        offset: Tuple[int, int]
            Zero-based (row, column) indices in the range [0, Rows) and
            [0, Columns), respectively, that specify the offset of the image
            region in the total pixel matrix. The ``(0, 0)`` coordinate is
            located at the center of the topleft hand pixel in the total pixel
            matrix.
        size: Tuple[int, int]
            Rows and columns of the requested image region
        channel_index: int, optional
            Zero-based index into channels along the direction defined by
            successive items of the appropriate DICOM attribute.
        focal_plane_index: int, optional
            Zero-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute.
            Values must be in the range [0, Total Pixel Matrix Focal Planes)

        Returns
        -------
        numpy.ndarray
            Three-dimensional pixel array of shape
            (Rows, Columns, Samples per Pixel) for the requested image region

        """
        matrix = self.get_total_pixel_matrix(
            channel_index=channel_index,
            focal_plane_index=focal_plane_index
        )

        row_start, col_start = offset
        rows, cols = size
        row_stop = row_start + rows
        col_stop = col_start + cols
        logger.debug(
            f'get region [{row_start}:{row_stop}, {col_start}:{col_stop}, :] '
            f'for channel {channel_index} and focal plane {focal_plane_index} '
            f'of image "{self._metadata.SOPInstanceUID}"'
        )
        return matrix[row_start:row_stop, col_start:col_stop, :]

    def get_slide_region(
        self,
        offset: Tuple[float, float],
        size: Tuple[float, float],
        channel_index: int = 0,
        focal_plane_index: int = 0
    ) -> np.ndarray:
        """Get slide region.

        Parameters
        ----------
        offset: Tuple[float, float]
            Zero-based (x, y) offset in the slide coordinate system in
            millimeter resolution. The ``(0.0, 0.0)`` coordinate is located at
            the origin of the slide (usually the slide corner).
        size: Tuple[float, float]
            Width and height of the requested slide region in millimeter unit
            along the X and Y axis of the slide coordinate system, respectively.
        channel_index: int, optional
            Zero-based index into channels along the direction defined by
            successive items of the appropriate DICOM attribute.
        focal_plane_index: int, optional
            Zero-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute.
            Values must be in the range [0, Total Pixel Matrix Focal Planes)

        Returns
        -------
        numpy.ndarray
            Three-dimensional pixel array of shape
            (Rows, Columns, Samples per Pixel) for the requested slide region

        Note
        ----
        The slide coordinate system is defined for the upright standing slide
        such that the X axis corresponds to the short side of the slide and the
        Y axis corresponds to the long side of the slide.
        The rows of the returned pixel array are parallel to the X axis of the
        slide coordinate system and the columns parallel to the Y axis of the
        slide coordinate system.

        """
        logger.debug(
            f'get region on slide of size {size} [mm] at offset {offset} [mm] '
            f'for channel {channel_index} and focal plane {focal_plane_index} '
            f'of image "{self._metadata.SOPInstanceUID}"'
        )
        matrix = self.get_total_pixel_matrix(
            channel_index=channel_index,
            focal_plane_index=focal_plane_index
        )

        pixel_indices = np.array([
            self.get_pixel_indices(offset=offset),
            self.get_pixel_indices(
                offset=(
                    offset[0] + size[0],
                    offset[1] + size[1],
                )
            )
        ])
        row_index = np.min(pixel_indices[:, 0])
        col_index = np.min(pixel_indices[:, 1])

        region_rows = np.abs(np.max(pixel_indices[:, 0]) - row_index)
        region_cols = np.abs(np.max(pixel_indices[:, 1]) - col_index)

        # Region may extend beyond the image's total pixel matrix
        col_overhang = abs(min([col_index, 0]))
        col_start = min([
            max([col_index, col_overhang]),
            matrix.shape[1] - 1,
        ])
        cols = min([
            region_cols - col_overhang - 1,
            matrix.shape[1] - col_start - 1,
        ])
        col_stop = col_start + cols
        region_col_start = int(col_overhang)
        region_col_stop = region_col_start + cols

        col_diff = (region_col_stop - region_col_start) - (col_stop - col_start)
        if col_diff != 0:
            raise ValueError(
                'Failed to get slide region: '
                'Could not determine the number of columns.'
            )

        row_overhang = abs(min([row_index, 0]))
        row_start = min([
            max([row_index, row_overhang]),
            matrix.shape[0] - 1,
        ])
        rows = min([
            region_rows - row_overhang - 1,
            matrix.shape[0] - row_start - 1,
        ])
        row_stop = row_start + rows
        region_row_start = int(row_overhang)
        region_row_stop = region_row_start + rows

        row_diff = (region_row_stop - region_row_start) - (row_stop - row_start)
        if row_diff != 0:
            raise ValueError(
                'Failed to get slide region: '
                'Could not determine the number of rows.'
            )

        region = np.zeros(
            (region_rows, region_cols, matrix.shape[2]),
            dtype=matrix.dtype
        )
        if self.metadata.SamplesPerPixel == 3:
            region += 255

        try:
            region[
                region_row_start:region_row_stop,
                region_col_start:region_col_stop
            ] = matrix[row_start:row_stop, col_start:col_stop, :]
        except ValueError as error:
            raise ValueError(f'Failed to get slide region: {error}')

        degrees = self.get_rotation()

        return rotate(region, angle=degrees)
