import itertools
import logging
from hashlib import sha256

import highdicom as hd
import numpy as np
from dicomweb_client import DICOMClient
from pydicom.dataset import Dataset

from dicomslide.matrix import TotalPixelMatrix
from dicomslide.tile import compute_frame_positions
from dicomslide.utils import encode_dataset

logger = logging.getLogger(__name__)


class TiledImage:

    """A tiled DICOM image.

    An instance of the class represents a tiled DICOM image instance and
    provides methods for convenient and efficient access of image metadata and
    pixel data from a DICOMweb server (or another source for which the
    :class:`dicomweb_client.DICOMClient` protocol has been implemented).

    Each image is associated with one or more :class:`TotalPixelMatrix`
    instances, one per optical path and focal plane.

    Examples
    --------
    >>> image = TiledImage(...)
    >>> print(image.metadata)  # pydicom.Dataset
    >>> print(image.metadata.BitsAllocated)
    >>> print(image.metadata.TotalPixelMatrixRows)
    >>> pixel_matrix = image.get_tota_pixel_matrix(optical_path_index=1)
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
        self._client = client
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
        self._optical_path_identifiers = tuple([
            str(item.OpticalPathIdentifier)
            for item in self._metadata.OpticalPathSequence
        ])
        self._number_of_optical_paths = len(self._optical_path_identifiers)

        iterator = itertools.product(
            range(1, self._number_of_optical_paths + 1),
            range(1, self._number_of_focal_planes + 1),
        )
        self._total_pixel_matrix_lut = {
            (optical_path_index, focal_plane_index): TotalPixelMatrix(
                client=self._client,
                image_metadata=self._metadata,
                optical_path_index=optical_path_index,
                focal_plane_index=focal_plane_index,
                max_frame_cache_size=max_frame_cache_size
            )
            for optical_path_index, focal_plane_index in iterator
        }

        encoded_dataset = encode_dataset(self._metadata)
        self._quickhash = sha256(encoded_dataset).hexdigest()

    def __hash__(self) -> str:
        return hash(self._quickhash)

    @property
    def metadata(self) -> Dataset:
        """pydicom.dataset.Dataset: Image metadata"""
        return self._metadata

    @property
    def num_optical_paths(self) -> int:
        """int: Number of optical paths"""
        return self._number_of_optical_paths

    def get_optical_path_index(self, optical_path_identifier: str) -> int:
        """Get index of an optical path.

        Parameters
        ----------
        optical_path_identifier: str
            Optical path identifier

        Returns
        -------
        int
            One-based index into optical paths along the direction defined by
            successive items of the Optical Path Sequence attribute. Values
            must be in the range [1, Number of Optical Paths].

        Raises
        ------
        ValueError
            When no optical path is found for `optical_path_identifier`

        """
        try:
            index = self._optical_path_identifiers.index(
                optical_path_identifier
            )
        except ValueError:
            raise ValueError(
                f'Image "{self._metadata.SOPInstanceUID}" does not have an '
                f'optical path with identifier "{optical_path_identifier}".'
            )
        return index + 1

    def get_optical_path_identifier(self, optical_path_index: int) -> str:
        """Get identifier of an optical path.

        Parameters
        ----------
        optical_path_index: int
            One-based index into optical paths along the direction defined by
            successive items of the Optical Path Sequence attribute. Values
            must be in the range [1, Number of Optical Paths].

        Returns
        -------
        str
            Optical path identifier

        Raises
        ------
        ValueError
            When no optical path is found for `optical_path_index`

        """
        try:
            return self._optical_path_identifiers[optical_path_index - 1]
        except IndexError:
            raise ValueError(
                f'Image "{self._metadata.SOPInstanceUID}" does not have an '
                f'optical path with index {optical_path_index}.'
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
            One-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute.
            Values must be in the range [1, Total Pixel Matrix Focal Planes]

        Raises
        ------
        ValueError
            When no focal plane is found for `focal_plane_offset`

        """
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
            One-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute.
            Values must be in the range [1, Total Pixel Matrix Focal Planes]

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
        optical_path_index: int = 1,
        focal_plane_index: int = 1
    ) -> TotalPixelMatrix:
        """Get total pixel matrix for a given optical path and focal plane.

        Parameters
        ----------
        optical_path_index: int, optional
            One-based index into optical paths along the direction defined by
            successive items of the Optical Path Sequence attribute. Values
            must be in the range [1, Number of Optical Paths].
        focal_plane_index: int, optional
            One-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute.
            Values must be in the range [1, Total Pixel Matrix Focal Planes]

        Returns
        -------
        dicomslide.TotalPixelMatrix
            Total Pixel Matrix

        """
        if (
            optical_path_index < 1 or
            optical_path_index > self.num_optical_paths
        ):
            raise ValueError(
                'Argument "optical_path_index" must be in range '
                f'[1, {self.num_optical_paths}], but it is '
                f'{optical_path_index}'
            )
        if (
            focal_plane_index < 1 or
            focal_plane_index > self.num_focal_planes
        ):
            raise ValueError(
                'Argument "focal_plane_index" must be in range '
                f'[1, {self.num_focal_planes}], but it is '
                f'{focal_plane_index}'
            )

        key = (optical_path_index, focal_plane_index)
        try:
            return self._total_pixel_matrix_lut[key]
        except KeyError:
            raise ValueError(
                'Could not find a Total Pixel Matrix for '
                f'Optical Path #{optical_path_index} and '
                f'Focal Plane #{focal_plane_index}.'
            )


# class TileIterator:

#     """Class for iterating over tiles of a tiled image.

#     Attributes
#     ----------
#     image: tumor_classification.image.TiledPlanarImage
#         DICOM image
#     padding: Tuple[int, int, int, int]
#         Number of pixels that should be padded to each frame at the left, top,
#         right, and bottom side.

#     """

#     def __init__(
#         self,
#         image: Image,
#         optical_path_index: int = 1,
#         focal_plane_index: int = 1,
#         padding: Union[int, Sequence[int]]
#     ) -> None:
#         """Construct object.

#         Parameters
#         ----------
#         image: tumor_classification.image.Image
#             DICOM image
#         optical_path_index: int, optional
#             One-based index into optical paths along the direction defined by
#             successive items of the Optical Path Sequence attribute. Values
#             must be in the range [1, Number of Optical Paths].
#         focal_plane_index: int, optional
#             One-based index into focal planes along depth direction from the
#             glass slide towards the coverslip in the slide coordinate system
#             specified by the Z Offset in Slide Coordinate System attribute.
#             Values must be in the range [1, Total Pixel Matrix Focal Planes]
#         padding: Union[int, Sequence[int]
#             Number of pixels that should be padded at the different sides of
#             the frame's pixel matrix. If an integer is provided, each side will
#             be padded with the same number of pixels. If a sequence of length
#             two is provided, then the first value applies to the left and right
#             sides and the second value to the top and bottom sides. If a
#             sequence of length four is provided, then the four values apply to
#             the left, top, right, and bottom side.

#         """
#         self.image = image
#         self._rows = int(self.image.metadata.Rows)
#         self._cols = int(self.image.metadata.Columns)

#         def check_pad_value(value: int) -> None:
#             if not isinstance(value, int):
#                 raise TypeError('Padding value must be an integer.')
#             if value < 0:
#                 raise ValueError(
#                     'Padding value must be either zero or a positive number.'
#                 )

#         if isinstance(padding, int):
#             check_pad_value(padding)
#             self.padding = (padding, padding, padding, padding)
#         elif isinstance(padding, Sequence):
#             [check_pad_value(value) for value in padding]
#             if len(padding) == 2:
#                 self.padding = (padding[0], padding[1], padding[0], padding[1])
#             elif len(padding) == 4:
#                 self.padding = tuple(padding)
#             else:
#                 raise ValueError(
#                     'When argument "padding" is a sequence, its length must '
#                     'be either 2 or 4.'
#                 )
#         else:
#             raise ValueError(
#                 'Argument "padding" must either be an integer or a sequence '
#                 'of integers.'
#             )

#         # Note sure we need to restrict usage in that way. The idea here is
#         # prevent padding to exceed the number of tile columns or rows
#         # to avoid that too retrieval of a large number of frames.
#         if self.padding[0] > self._cols:
#             raise ValueError(
#                 'Padding at the left side exceeds number of columns.'
#             )
#         if self.padding[2] > self._cols:
#             raise ValueError(
#                 'Padding at the right side exceeds number of columns.'
#             )
#         if self.padding[1] > self._rows:
#             raise ValueError(
#                 'Padding at the top side exceeds number of rows.'
#             )
#         if self.padding[3] > self._rows:
#             raise ValueError(
#                 'Padding at the bottom side exceeds number of rows.'
#             )

#         (
#             matrix_positions,
#             slide_offsets,
#             optical_path_indices,
#             focal_plane_indices,
#         ) = compute_frame_positions(self.image.metadata)
#         selection_index = np.logical_and(
#             optical_path_indices == optical_path_index,
#             focal_plane_indices == focal_plane_index
#         )
#         self._tile_positions = matrix_positions[selection_index, :]
#         # We want to iterate first along the row direction (left -> right) and
#         # then along the column direction (top -> bottom) for improved caching
#         # efficiency. By ensuring that that two subsequent frame requests will
#         # be made for neighboring tiles, pixels can be shared accross requests
#         # when padding is applied, resulting in a lower overall number of
#         # requests.
#         tile_grid_indices = np.column_stack([
#             np.floor((self._tile_positions[:, 1] - 1) / self._rows),
#             np.floor((self._tile_positions[:, 0] - 1) / self._cols),
#         ]).astype(int)
#         sort_index = np.lexsort([
#             tile_grid_indices[:, 1],
#             tile_grid_indices[:, 0]
#         ])
#         frame_indices = np.arange(0, int(self.image.NumberOfFrames))
#         frame_indices = frame_indices[selection_index]
#         self._sorted_frame_indices = frame_indices[sort_index]
#         self._current_index = 0

#     def __len__(self):
#         return len(self._sorted_frame_indices)

#     def __iter__(self):
#         return self

#     def __next__(self) -> np.ndarray:
#         if self._current_index >= len(self):
#             raise StopIteration
#         index = int(self._current_index)
#         self._current_index += 1
#         return self[index]

#     def __getitem__(self, key: Union[int, List[int], slice]) -> np.ndarray:
#         indices = []
#         if isinstance(key, tuple):
#             if isinstance(key[0], int):
#                 indices.extend([k for k in key])
#             elif isinstance(key[0], slice):
#                 raise ValueError('Slice index must be one-dimensional.')
#             else:
#                 raise TypeError(
#                     'Index must be an integer, a list of integers, or a slice.'
#                 )
#         elif isinstance(key, list):
#             if isinstance(key[0], int):
#                 indices.extend([k for k in key])
#             else:
#                 raise TypeError(
#                     'Index must be an integer, a list of integers, or a slice.'
#                 )
#         elif isinstance(key, int):
#             indices.append(key)
#         else:
#             raise TypeError(
#                 'Index must be an integer, a list of integers, or a slice.'
#             )

#         if len(indices) > 1:
#             regions = [
#                 self._get_region_for_tile(index=index)
#                 for index in indices
#             ]
#             return np.stack(regions)
#         else:
#             return self._get_region_for_tile(index=indices[0])

#     @property
#     def tile_size(self) -> Tuple[int, int]:
#         """Tuple[int, int]: Number of pixel rows and columns of a tile."""
#         return (self._rows, self._cols)

#     @property
#     def region_size(self) -> Tuple[int, int]:
#         """Tuple[int, int]: Number of pixel rows and columns of a region."""
#         left, top, right, bottom = self.padding
#         return (self._rows + top + bottom, self._cols + left + right)

#     @property
#     def region_shape(self) -> Tuple[int, int, int]:
#         """Tuple[int, int]: Number of pixel rows, number of pixel columns, and
#         number of samples per pixel of a region."""
#         return (*self.region_size, self.image.metadata.SamplesPerPixel)

#     def extract_tile_from_region(self, region: np.ndarray) -> np.ndarray:
#         """Extract pixels of a tile from a region.

#         Parameters
#         ----------
#         region: numpy.ndarray
#             Image region containing the pixels of the tile of interest

#         Returns
#         -------
#         numpy.ndarray
#             Pixel matrix of tile

#         """
#         left, top, right, bottom = self.padding
#         return region[left:-right, top:-bottom, :]

#     def _get_region_for_tile(self, index: int) -> np.ndarray:
#         """Get region for a tile.

#         The region includes the pixels of the frame for the tile as well as
#         parts of additional frames to the left, top, right, and bottom of the
#         frame. How many pixels of the other frames are included in the region
#         depends on the value of the `padding` attribute.

#         Parameters
#         ----------
#         index: int
#             Zero-based index of the tile (this is **not** necessarily the index
#             of the frame)

#         Returns
#         -------
#         numpy.ndarray
#             Image region including pixels from the tile and potentially
#             parts of neighboring tiles

#         """  # noqa: E501
#         frame_index = self._sorted_frame_indices[index]
#         total_col_start, total_row_start = (
#             self._tile_positions[frame_index, :] - 1
#         )
#         left, top, right, bottom = self.padding
#         tile_rows, tile_cols = self.tile_size
#         total_rows, total_cols = self.image.pixel_array.shape[:2]

#         region_rows, region_cols = self.region_size
#         region = np.zeros(self.region_shape, dtype=self.image.pixel_array.dtype)

#         if total_col_start > 0:
#             region_col_start = 0
#             adjusted_total_col_start = total_col_start - left
#         else:
#             region_col_start = left
#             adjusted_total_col_start = total_col_start
#         col_diff = total_cols - (total_col_start + tile_cols)
#         if col_diff <= 0:
#             region_col_end = region_cols
#             adjusted_total_col_end = total_col_start + tile_cols + right
#         else:
#             if right > 0:
#                 region_col_end = -right
#             else:
#                 region_col_end = None
#             adjusted_total_col_end = total_col_start + tile_cols

#         if total_row_start > 0:
#             region_row_start = 0
#             adjusted_total_row_start = total_row_start - top
#         else:
#             region_row_start = top
#             adjusted_total_row_start = total_row_start
#         row_diff = total_rows - (total_row_start + tile_rows)
#         if row_diff <= 0:
#             region_row_end = region_rows
#             adjusted_total_row_end = total_row_start + tile_rows + bottom
#         else:
#             if bottom > 0:
#                 region_row_end = -bottom
#             else:
#                 region_row_end = None
#             adjusted_total_row_end = total_row_start + tile_rows

#         adjusted_region = self.image.pixel_array[
#             adjusted_total_row_start:adjusted_total_row_end,
#             adjusted_total_col_start:adjusted_total_col_end,
#             :
#         ]
#         region[
#             region_row_start:region_row_end,
#             region_col_start:region_col_end,
#             :
#         ] = adjusted_region

#         return region
