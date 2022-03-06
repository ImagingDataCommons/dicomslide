import collections
import itertools
import logging
from typing import (
    Dict,
    Sequence,
    Tuple,
    Union
)

import highdicom as hd
import numpy as np
from dicomweb_client import DICOMClient
from pydicom.dataset import Dataset
from pydicom.uid import UID
from dicomslide.tile import (
    assemble_total_pixel_matrix,
    compute_tile_positions,
    is_tiled_image
)

logger = logging.getLogger(__name__)


def _determine_transfer_syntax(frame: bytes) -> UID:
    """Determine the transfer syntax of a frame.

    Parameters
    ----------
    frame: bytes
        Frame item of the Pixel Data element of an image

    Returns
    -------
    pydicom.uid.UID
        UID of transfer syntax

    """

    def is_jpeg(frame):
        start_marker = b"\xFF\xD8"
        end_markers = (b"\xFF\xD9", b"\xFF\xD9\x00")  # may be zero padded
        if frame.startswith(start_marker):
            if any([frame.endswith(marker) for marker in end_markers]):
                return True
        return False

    def is_jpeg2000(frame):
        start_marker = b"\xFF\x4F"
        end_markers = (b"\xFF\xD9", b"\xFF\xD9\x00")  # may be zero padded
        if frame.startswith(start_marker):
            if any([frame.endswith(marker) for marker in end_markers]):
                return True
        return False

    def is_jpegls(frame):
        start_markers = (b"\xFF\xD8\xFF\xF7", b"\xFF\xD8\xFF\xE8")
        end_markers = (b"\xFF\xD9", b"\xFF\xD9\x00")  # may be zero padded
        if any([frame.startswith(marker) for marker in start_markers]):
            if any([frame.endswith(marker) for marker in end_markers]):
                return True
        return False

    # We assume that frames are either JPEG, JPEG-LS, or JPEG2000 compressed
    if is_jpegls(frame):
        # Needs to be checked before JPEG because they share SOI and EOI marker
        return UID("1.2.840.10008.1.2.4.80")
    elif is_jpeg(frame):
        return UID("1.2.840.10008.1.2.4.50")
    elif is_jpeg2000(frame):
        return UID("1.2.840.10008.1.2.4.90")
    else:
        return UID("1.2.840.10008.1.2.1")


class TotalPixelMatrix:

    """A Total Pixel Matrix.

    The class exposes a NumPy-like interface to index into a total pixel matrix
    of a tiled image, where each tile is encoded as a separate frame.
    Instances of the class walk and quack like NumPy arrays and can be indexed
    accordingly. When the caller indexes instances of the class, the
    corresponding image frames are dynamically retrieved from a DICOM store and
    decoded and the individual tiles are stitched together to form the
    requested image region.

    Examples
    --------
    >>> pixel_array = TotalPixelMatrix(...)
    >>> print(pixel_array.dtype)
    >>> print(pixel_array.ndim)
    >>> print(pixel_array.shape)
    >>> print(pixel_array[:256, 256:512, :])  # numpy.ndarray

    Warning
    -------
    The total pixel matrix may be very large and indexing the row or column
    dimension with ``:`` may consume a lot of time and memory.

    """

    def __init__(
        self,
        client: DICOMClient,
        image_metadata: Dataset,
        optical_path_index: int = 1,
        focal_plane_index: int = 1,
        max_frame_cache_size: int = 9,
        correct_color: bool = True
    ) -> None:
        """Construct object.

        Parameters
        ----------
        client: dicomweb_client.api.DICOMClient
            DICOMweb client
        image_metadata: pydicom.dataset.Dataset
            Metadata of a tiled DICOM image
        optical_path_index: int, optional
            One-based index into optical paths along the direction defined by
            successive items of the Optical Path Sequence attribute. Values
            must be in the range [1, Number of Optical Paths].
        focal_plane_index: int, optional
            One-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute.
            Values must be in the range [1, Total Pixel Matrix Focal Planes]
        max_frame_cache_size: int, optional
            Maximum number of frames that should be cached to avoid repeated
            retrieval requests
        correct_color: bool, optional
            Whether pixel values should be color corrected

        """
        self._client = client
        self._metadata = image_metadata
        if not is_tiled_image(image_metadata):
            raise ValueError('Image is not tiled.')

        num_optical_paths = len(self._metadata.OpticalPathSequence)
        if optical_path_index < 1 or optical_path_index > num_optical_paths:
            raise ValueError(
                'Argument "optical_path_index" must be in range '
                f'[1, {self.num_optical_paths}].'
            )
        num_focal_planes = getattr(
            self._metadata,
            'TotalPixelMatrixFocalPlanes',
            1
        )
        if focal_plane_index < 1 or focal_plane_index > num_focal_planes:
            raise ValueError(
                'Argument "focal_plane_index" must be in range '
                f'[1, {self.num_focal_planes}].'
            )

        self._n = int(image_metadata.NumberOfFrames)
        self._rows = int(image_metadata.Rows)
        self._cols = int(image_metadata.Columns)
        (
            matrix_positions,
            slide_offsets,
            optical_path_indices,
            focal_plane_indices,
        ) = compute_tile_positions(image_metadata)
        self._tile_positions = matrix_positions[
            np.logical_and(
                optical_path_indices == optical_path_index,
                focal_plane_indices == focal_plane_index
            ),
            :
        ]
        self._tile_grid_indices = np.column_stack([
            np.floor((self._tile_positions[:, 1] - 1) / self._rows),
            np.floor((self._tile_positions[:, 0] - 1) / self._cols),
        ]).astype(int)

        self._cache = collections.OrderedDict()
        if max_frame_cache_size < 0:
            raise ValueError(
                'Cache size must be at an unsigned integer value.'
            )
        self._max_frame_cache_size = int(max_frame_cache_size)

        if self._metadata.SamplesPerPixel == 3 and correct_color:
            if hasattr(self._metadata.OpticalPathSequence[0], 'ICCProfile'):
                icc_profile = self._metadata.OpticalPathSequence[0].ICCProfile
                color_manager = hd.color.ColorManager(icc_profile)
                self._transform_fn = color_manager.transform_frame
            else:
                logger.warning(
                    'color image does not contain an ICC profile - '
                    'pixel values will not be color corrected'
                )
                self._transform_fn = lambda x: x
        else:
            self._transform_fn = lambda x: x

    def _decode_frame(
        self,
        frame: bytes,
        transfer_syntax_uid: str
    ) -> np.ndarray:
        """Decode a frame.

        Parameters
        ----------
        frame: bytes
            Frame item of the Pixel Data element of an image
        transfer_syntax_uid: str
            UID of the transfer syntax

        Returns
        -------
        numpy.ndarray
            Decoded frame pixel array

        """
        metadata = self._metadata
        return hd.frame.decode_frame(
            frame,
            transfer_syntax_uid=transfer_syntax_uid,
            rows=metadata.Rows,
            columns=metadata.Columns,
            samples_per_pixel=metadata.SamplesPerPixel,
            bits_allocated=metadata.BitsAllocated,
            bits_stored=metadata.BitsStored,
            photometric_interpretation=metadata.PhotometricInterpretation,
            pixel_representation=metadata.PixelRepresentation,
            planar_configuration=getattr(metadata, 'PlanarConfiguration', None)
        )

    def _retrieve_and_decode_frames(
        self,
        frame_indices: Sequence[int]
    ) -> Dict[int, np.ndarray]:
        """Retrieve and decode frames.

        Parameters
        ----------
        frame_indices: Sequence[int]
            Zero-based indices of frames that should be retrieved

        Returns
        -------
        Dict[int, np.ndarray]
            Mapping of zero-based frame index to frame pixel array

        """
        # Retrieve frames that have not yet been cached
        selected_frame_numbers = []
        for index in frame_indices:
            if index not in self._cache:
                selected_frame_numbers.append(index + 1)

        if len(selected_frame_numbers) > 0:
            frames = self._client.retrieve_instance_frames(
                study_instance_uid=self._metadata.StudyInstanceUID,
                series_instance_uid=self._metadata.SeriesInstanceUID,
                sop_instance_uid=self._metadata.SOPInstanceUID,
                frame_numbers=selected_frame_numbers,
                media_types=(
                    ("application/octet-stream", "*"),
                    ("image/jpeg", "1.2.840.10008.1.2.4.50"),
                    ("image/jls", "1.2.840.10008.1.2.4.80"),
                    ("image/jls", "1.2.840.10008.1.2.4.81"),
                    ("image/jp2", "1.2.840.10008.1.2.4.90"),
                    ("image/jp2", "1.2.840.10008.1.2.4.91"),
                    ("image/jpx", "1.2.840.10008.1.2.4.92"),
                    ("image/jpx", "1.2.840.10008.1.2.4.93"),
                )
            )
            # Decode and cache retrieved frames
            for i, number in enumerate(selected_frame_numbers):
                frame_item = frames[i]
                array = self._decode_frame(
                    frame=frame_item,
                    transfer_syntax_uid=_determine_transfer_syntax(frame_item)
                )
                array = self._transform_fn(array)
                if self._metadata.SamplesPerPixel == 1 and array.ndim == 2:
                    array = array[..., np.newaxis]
                index = number - 1
                self._cache[index] = array

        # Get cached frames
        pixel_array_mapping = {
            index: self._cache[index]
            for index in frame_indices
        }

        # Invalidate cache
        cache_diff = len(self._cache) - self._max_frame_cache_size
        if cache_diff > 0:
            for _ in range(cache_diff):
                self._cache.popitem(last=False)

        return pixel_array_mapping

    @property
    def dtype(self) -> np.dtype:
        """numpy.dtype: Data type"""
        return np.dtype(f'uint{self._metadata.BitsAllocated}')

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Tuple[int, int, int]: Image Rows, Columns, and Samples per Pixel"""
        return (
            self._metadata.TotalPixelMatrixRows,
            self._metadata.TotalPixelMatrixColumns,
            self._metadata.SamplesPerPixel
        )

    @property
    def ndim(self) -> int:
        """int: Number of dimensions"""
        return len(self.shape)

    def _get_frame_index(self, tile_grid_index: Tuple[int, int]) -> int:
        """Get index of a frame.

        Parameters
        ----------
        tile_grid_position: Tuple[int, int]
            Zero-based (row, column) index of a tile in the tile grid

        Returns
        -------
        int
            Zero-based index of the frame in the pixel data

        """
        matches = np.logical_and(
            self._tile_grid_indices[:, 0] == tile_grid_index[0],
            self._tile_grid_indices[:, 1] == tile_grid_index[1]
        )
        return int(np.where(matches)[0][0])

    def __getitem__(
        self,
        key: Tuple[Union[slice, int], Union[slice, int], Union[slice, int]]
    ) -> np.ndarray:
        if not isinstance(key, tuple) or len(key) != 3:
            raise ValueError(
                'Slice Index must be three-dimensional to specify the extent '
                'of the image region along the column direction '
                '(top to bottom), the row direction (left to right), and '
                'the channel direction.'
            )

        # Rows
        if isinstance(key[0], int):
            row_start = key[0]
            row_stop = key[0] + 1
        elif isinstance(key[0], slice):
            if key[0].start is None:
                row_start = 0
            else:
                row_start = max(
                    0,
                    key[0].start
                )
            if key[0].stop is None:
                row_stop = self._metadata.TotalPixelMatrixRows
            else:
                row_stop = min(
                    key[0].stop,
                    self._metadata.TotalPixelMatrixRows
                )
        else:
            raise TypeError(
                'Item #0 of argument "key" must be an integer or a slice.'
            )
        row_start_tile_fraction = row_start / self._rows
        row_start_tile_index = int(np.floor(row_start_tile_fraction))
        row_stop_tile_fraction = row_stop / self._rows
        row_stop_tile_index = int(np.ceil(row_stop_tile_fraction))
        row_tile_range = list(range(row_start_tile_index, row_stop_tile_index))
        region_row_start = int(
            np.floor(
                (row_start_tile_fraction - row_start_tile_index) * self._rows
            )
        )
        region_row_stop = -1 * int(
            np.ceil(
                (row_stop_tile_index - row_stop_tile_fraction) * self._rows
            )
        )
        if region_row_stop == 0:
            region_row_stop = None

        # Columns
        if isinstance(key[1], int):
            col_start = key[1]
            col_stop = key[1] + 1
        elif isinstance(key[1], slice):
            if key[1].start is None:
                col_start = 0
            else:
                col_start = max(
                    0,
                    key[1].start
                )
            if key[1].stop is None:
                col_stop = self._metadata.TotalPixelMatrixColumns
            else:
                col_stop = min(
                    key[1].stop,
                    self._metadata.TotalPixelMatrixColumns
                )
        else:
            raise TypeError(
                'Item #1 of argument "key" must be an integer or a slice.'
            )
        col_start_tile_fraction = col_start / self._cols
        col_start_tile_index = int(np.floor(col_start_tile_fraction))
        col_stop_tile_fraction = col_stop / self._cols
        col_stop_tile_index = int(np.ceil(col_stop_tile_fraction))
        col_tile_range = list(range(col_start_tile_index, col_stop_tile_index))
        region_col_start = int(
            np.floor(
                (col_start_tile_fraction - col_start_tile_index) * self._cols
            )
        )
        region_col_stop = -1 * int(
            np.ceil(
                (col_stop_tile_index - col_stop_tile_fraction) * self._cols
            )
        )
        if region_col_stop == 0:
            region_col_stop = None

        # Samples
        if isinstance(key[2], int):
            sample_start = key[2]
            sample_stop = key[2] + 1
        elif isinstance(key[2], slice):
            if key[2].start is None:
                sample_start = 0
            else:
                sample_start = max(
                    0,
                    key[2].start
                )
            if key[2].stop is None:
                sample_stop = self._metadata.SamplesPerPixel
            else:
                sample_stop = max(
                    key[2].stop,
                    self._metadata.SamplesPerPixel
                )
        else:
            raise TypeError(
                'Item #2 of argument "key" must be an integer or a slice.'
            )

        frame_position_mapping = {}
        for (r, c) in itertools.product(row_tile_range, col_tile_range):
            frame_index = self._get_frame_index((r, c))
            # (Column, Row) position of the frame in the region pixel matrix.
            # The top left frame is located at (1, 1).
            # These tile positions will subsequently be passed to the
            # assemble_total_pixel_matrix() function, which expects the tile
            # positions to be provided according the specification of the DICOM
            # Column/Row Position In Total Image Pixel Matrix attributes.
            frame_position_mapping[frame_index] = (
                (
                    self._tile_positions[frame_index, 0]
                    - (col_start_tile_index * self._cols)
                ),
                (
                    self._tile_positions[frame_index, 1]
                    - (row_start_tile_index * self._rows)
                ),
            )

        frame_indices = [key for key in frame_position_mapping.keys()]
        frame_array_mapping = self._retrieve_and_decode_frames(frame_indices)
        tiles = [frame_array_mapping[i] for i in frame_indices]
        tile_positions = np.array([
            frame_position_mapping[i] for i in frame_indices
        ])
        if tile_positions.shape[0] > 0:
            extended_region = assemble_total_pixel_matrix(
                tiles=tiles,
                tile_positions=tile_positions,
                total_pixel_matrix_columns=int(
                    np.max(tile_positions[:, 0]) - 1 + self._cols
                ),
                total_pixel_matrix_rows=int(
                    np.max(tile_positions[:, 1]) - 1 + self._rows
                )
            )
            region = extended_region[
                region_row_start:region_row_stop,
                region_col_start:region_col_stop,
                sample_start:sample_stop
            ]
            # Return a copy rather than a view to provide a continous block of
            # memory without the extra bytes that are no longer needed.
            return region.copy()
        else:
            shape = []
            if len(row_tile_range) == 0:
                shape.append(0)
            else:
                if region_row_stop < 0:
                    adjusted_region_row_stop = np.sum([
                        self._metadata.TotalPixelMatrixRows,
                        region_row_stop
                    ])
                else:
                    adjusted_region_row_stop = region_row_stop
                region_row_diff = adjusted_region_row_stop - region_row_start
                shape.append(max([0, region_row_diff]))

            if len(col_tile_range) == 0:
                shape.append(0)
            else:
                if region_col_stop < 0:
                    adjusted_region_col_stop = np.sum([
                        self._metadata.TotalPixelMatrixColumns,
                        region_col_stop
                    ])
                else:
                    adjusted_region_col_stop = region_col_stop
                region_col_diff = adjusted_region_col_stop - region_col_start
                shape.append(max([0, region_col_diff]))

            if sample_stop < 0:
                adjusted_sample_stop = np.sum([
                    self._metadata.SamplesPerPixel,
                    sample_stop
                ])
            else:
                adjusted_sample_stop = sample_stop
            sample_diff = adjusted_sample_stop - sample_start
            shape.append(max([0, sample_diff]))

            return np.zeros(shape, dtype=self.dtype)
