import itertools
import logging
from collections import OrderedDict
from typing import (
    Callable,
    Dict,
    Optional,
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
    compute_frame_positions,
    is_tiled_image
)

logger = logging.getLogger(__name__)


def _determine_transfer_syntax(
    frame: bytes,
    metadata: Dataset
) -> UID:
    """Determine the transfer syntax of a frame.

    Parameters
    ----------
    frame: bytes
        Frame item of the Pixel Data element of an image
    metadata: pydicom.Dataset
        Image metadata

    Returns
    -------
    pydicom.uid.UID
        UID of transfer syntax

    Warning
    -------
    The function makes the assumption that the frame was encoded using
    using a JPEG, JPEG 2000, or JPEG-LS compression codec or encoded natively
    (uncompressed). If another compression codec was used, the function will
    fail to detec the correct transfer syntax.

    """
    def is_lossy_compressed(metadata: Dataset) -> bool:
        lossy_image_compression = metadata.get('LossyImageCompression', '00')
        return lossy_image_compression == '01'

    def do_markers_match(
        frame: bytes,
        start_markers: Tuple[bytes, ...],
        end_markers: Tuple[bytes, ...]
    ) -> bool:
        if any([frame.startswith(marker) for marker in start_markers]):
            if any([frame.endswith(marker) for marker in end_markers]):
                return True
        return False

    def is_jpeg(frame: bytes) -> bool:
        start_markers = (b"\xFF\xD8", )
        end_markers = (b"\xFF\xD9", b"\xFF\xD9\x00")  # may be zero padded
        return do_markers_match(frame, start_markers, end_markers)

    def is_jpeg2000(frame: bytes) -> bool:
        start_markers = (
            b"\x00\x00\x00\x0C\x6A\x50\x20\x20\x0D\x0A\x87\x0A",  # jp2 (boxed)
            b"\xff\x4f\xff\x51",  # j2k
        )
        end_markers = (b"\xFF\xD9", b"\xFF\xD9\x00")  # may be zero padded
        return do_markers_match(frame, start_markers, end_markers)

    def is_jpegls(frame: bytes) -> bool:
        start_markers = (b"\xFF\xD8\xFF\xF7", b"\xFF\xD8\xFF\xE8")
        end_markers = (b"\xFF\xD9", b"\xFF\xD9\x00")  # may be zero padded
        return do_markers_match(frame, start_markers, end_markers)

    if is_jpegls(frame):
        # Needs to be checked before JPEG because they share SOI and EOI marker
        if not is_lossy_compressed(metadata):
            return UID("1.2.840.10008.1.2.4.80")
        return UID("1.2.840.10008.1.2.4.81")
    elif is_jpeg(frame):
        if not is_lossy_compressed(metadata):
            raise ValueError(
                "Transfer syntax determined from value of frame item and "
                "image metadata do not match."
            )
        return UID("1.2.840.10008.1.2.4.50")
    elif is_jpeg2000(frame):
        if not is_lossy_compressed(metadata):
            return UID("1.2.840.10008.1.2.4.90")
        return UID("1.2.840.10008.1.2.4.91")
    else:
        return UID("1.2.840.10008.1.2.1")


class TotalPixelMatrix:

    """Total Pixel Matrix.

    The class exposes a NumPy-like interface to index into a total pixel matrix
    of a tiled image, where each tile is encoded as a separate frame.
    Instances of the class walk and quack like NumPy arrays and can be indexed
    accordingly. When the caller indexes instances of the class, the
    corresponding image frames are dynamically retrieved from a DICOM store and
    decoded.

    A notable difference to NumPy array indexing is that a one-dimensional
    index returns an individual tile of the total pixel matrix (i.e., a 2D
    array) rather than an individual row of the total pixel matrix (i.e., a 1D
    array).

    The caller can index instances of the class either using one-dimensional
    tile indices into the flattened list of tiles in the total pixel matrix to
    get individual tiles or three-dimensional pixel indices (rows, columns, and
    samples) into the total pixel matrix to get a continous region spanning one
    or more tiles.

    Examples
    --------
    >>> matrix = TotalPixelMatrix(...)
    >>> print(matrix.dtype)
    >>> print(matrix.ndim)
    >>> print(matrix.shape)
    >>> print(matrix.size)
    >>> region = matrix[:256, 256:512, :]
    >>> print(len(matrix))
    >>> tile = matrix[0]
    >>> tile = matrix[matrix.get_tile_index(2, 4)]
    >>> tiles = matrix[[0, 1, 2, 5, 6, 7]]
    >>> tiles = matrix[2:6]

    Warning
    -------
    The total pixel matrix may be very large and indexing the row or column
    dimension with ``:`` may consume a lot of time and memory.

    """

    def __init__(
        self,
        client: DICOMClient,
        image_metadata: Dataset,
        channel_index: int = 0,
        focal_plane_index: int = 0,
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
        channel_index: int, optional
            Zero-based index into channels along the direction defined by
            successive items of the appropriate DICOM attribute(s).
        focal_plane_index: int, optional
            Zero-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute.
            Values must be in the range [0, Total Pixel Matrix Focal Planes).
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

        self._n = int(image_metadata.NumberOfFrames)
        self._rows = int(image_metadata.Rows)
        self._cols = int(image_metadata.Columns)
        (
            matrix_positions,
            slide_offsets,
            channel_indices,
            focal_plane_indices,
        ) = compute_frame_positions(image_metadata)

        num_channels = len(np.unique(channel_indices))
        if channel_index < 0 or channel_index >= num_channels:
            raise ValueError(
                'Argument "channel_index" must be a zero-based index '
                f'in range [0, {num_channels}).'
            )
        num_focal_planes = len(np.unique(focal_plane_indices))
        if focal_plane_index < 0 or focal_plane_index >= num_focal_planes:
            raise ValueError(
                'Argument "focal_plane_index" must be a zero-based index '
                f'in range [0, {num_focal_planes}).'
            )

        frame_selection_index = np.logical_and(
            channel_indices == channel_index,
            focal_plane_indices == focal_plane_index
        )
        self._tile_positions = matrix_positions[frame_selection_index, :]
        self._tile_grid_positions = np.column_stack([
            np.floor((self._tile_positions[:, 0]) / self._rows),
            np.floor((self._tile_positions[:, 1]) / self._cols),
        ]).astype(int)
        tile_sort_index = np.lexsort([
            self._tile_grid_positions[:, 0],
            self._tile_grid_positions[:, 1]
        ])
        self._tile_grid_rows = int(np.max(self._tile_grid_positions[:, 0])) + 1
        self._tile_grid_cols = int(np.max(self._tile_grid_positions[:, 1])) + 1
        self._num_frames = int(self._metadata.NumberOfFrames)
        frame_indices = np.arange(self._num_frames)
        self._frame_indices = frame_indices[frame_selection_index]
        self._sorted_frame_indices = self._frame_indices[tile_sort_index]
        self._current_index = 0

        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        if max_frame_cache_size < 0:
            raise ValueError(
                'Cache size must be at an unsigned integer value.'
            )
        self._max_frame_cache_size = int(max_frame_cache_size)

        self._transform_fn: Union[Callable[[np.ndarray], np.ndarray], None]
        if self._metadata.SamplesPerPixel == 3 and correct_color:
            if (
                hasattr(self._metadata, 'OpticalPathSequence') and
                hasattr(self._metadata.OpticalPathSequence[0], 'ICCProfile')
            ):
                icc_profile = self._metadata.OpticalPathSequence[0].ICCProfile
                color_manager = hd.color.ColorManager(icc_profile)
                self._transform_fn = color_manager.transform_frame
            else:
                logger.warning(
                    f'color image "{self._metadata.SOPInstanceUID}" does not '
                    'contain an ICC profile - '
                    'pixel values will not be color corrected'
                )
                self._transform_fn = None
        else:
            self._transform_fn = None

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
            frame_number = index + 1
            if frame_number > int(self._metadata.NumberOfFrames):
                raise ValueError(
                    f'Cannot retrieve frame #{index}. '
                    f'Image contains only n={self._num_frames} frames.'
                )
            if index not in self._cache:
                selected_frame_numbers.append(frame_number)
            else:
                logger.debug(f'reuse cached frame {frame_number}')

        if len(selected_frame_numbers) > 0:
            logger.debug(f'retrieve frames {selected_frame_numbers}')
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
                transfer_syntax_uid = _determine_transfer_syntax(
                    frame_item,
                    metadata=self._metadata
                )
                logger.debug(
                    f'decode frame {number} with transfer syntax '
                    f'"{transfer_syntax_uid}"'
                )
                array = self._decode_frame(
                    frame=frame_item,
                    transfer_syntax_uid=transfer_syntax_uid
                )
                if self._transform_fn is not None:
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
    def num_tiles(self) -> int:
        """int: Number of tiles"""
        return int(np.product(self.tile_grid_shape))

    @property
    def tile_grid_shape(self) -> Tuple[int, int]:
        """Tuple[int, int]: Number of tiles along the column (top to bottom)
        and row (left to right) direction of the tile grid

        """
        return (self._tile_grid_rows, self._tile_grid_cols)

    @property
    def tile_size(self) -> int:
        """int: Size of an invidual tile (rows x columns x samples)"""
        return int(np.product(self.tile_shape))

    @property
    def tile_shape(self) -> Tuple[int, int, int]:
        """Tuple[int, int, int]: Number of pixel rows, pixel columns, and
        samples per pixel of an individual tile

        """
        return (
            self._rows,
            self._cols,
            int(self._metadata.SamplesPerPixel)
        )

    @property
    def tile_grid_positions(self) -> np.ndarray:
        """numpy.ndarray: Two-dimensional array of integer values representing
        the grid positions of individual tiles in the tile grid

        """
        return self._tile_grid_positions

    def get_tile_grid_position(self, index: int) -> Tuple[int, int]:
        """Get position of a tile in the tile grid.

        Parameters
        ----------
        index: int
            Zero-based index of the tile in the flattened total pixel matrix

        Returns
        -------
        Tuple[int, int]
            Zero-based (row, column) index of a tile in the tile grid

        """
        position = self._tile_grid_positions[index, :]
        return (int(position[0]), int(position[1]))

    @property
    def tile_positions(self) -> np.ndarray:
        """numpy.ndarray: Two-dimensional array of integer values representing
        the positions of individual tiles in the total pixel matrix, i.e., the
        offsets from the ``(0, 0)`` origin of the total pixel matrix at the top
        lefthand pixel

        """
        return self._tile_positions

    def get_tile_position(self, index: int) -> Tuple[int, int]:
        """Get position of a tile.

        Parameters
        ----------
        index: int
            Zero-based index of the tile in the flattened total pixel matrix

        Returns
        -------
        Tuple[int, int]
            Zero-based (row, column) offset of a tile in the total pixel matrix

        """
        r, c = self._tile_positions[index]
        return (int(r), int(c))

    def get_tile_bounding_box(self, index: int) -> Tuple[
        Tuple[int, int], Tuple[int, int]
    ]:
        """Get the bounding box of a tile.

        Parameters
        ----------
        index: int
            Tile index

        Returns
        -------
        offset: Tuple[int, int]
            Zero-based (row, column) pixel indices in the total pixel matrix
        size: Tuple[int, int]
            Height (rows) and width (columns) of the tile

        """
        r, c = self._tile_positions[index]
        return (
            (r, c),
            (self._rows, self._cols),
        )

    @property
    def size(self) -> int:
        """int: Size (rows x columns x samples)"""
        return int(np.product(self.shape))

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Tuple[int, int, int]: Rows, Columns, and Samples per Pixel"""
        return (
            int(self._metadata.TotalPixelMatrixRows),
            int(self._metadata.TotalPixelMatrixColumns),
            int(self._metadata.SamplesPerPixel)
        )

    @property
    def ndim(self) -> int:
        """int: Number of dimensions"""
        return len(self.shape)

    def __len__(self) -> int:
        """Determine the number of tiles.

        Returns
        -------
        int
            Number of tiles

        """
        return self._tile_positions.shape[0]

    def __iter__(self):
        """Iterate over tiles."""
        self._current_index = 0
        return self

    def __next__(self) -> np.ndarray:
        """Serve the next tile.

        Returns
        -------
        numpy.ndarray
            3D tile array (rows x columns x samples)

        """
        if self._current_index >= len(self):
            raise StopIteration
        index = int(self._current_index)
        self._current_index += 1
        return self[index]

    def get_tile_index(self, position: Tuple[int, int]) -> int:
        """Get index of a tile.

        Parameters
        ----------
        position: Tuple[int, int]
            Zero-based (row, column) index of a tile in the tile grid

        Returns
        -------
        int
            Zero-based index of the tile in the flattened total pixel matrix

        """
        matches = np.logical_and(
            self._tile_grid_positions[:, 0] == position[0],
            self._tile_grid_positions[:, 1] == position[1]
        )
        if matches.shape[0] == 0:
            raise IndexError(f'Could not find a tile at position {position}.')
        return int(np.where(matches)[0][0])

    def _read_region(
        self,
        key: Tuple[Union[slice, int], Union[slice, int], Union[slice, int]],
    ) -> np.ndarray:
        """Read a continous region of pixels from one or more tiles.

        Retrieves one or more frames, decodes them, and stitches them together
        to form the requested region.

        Parameters
        ----------
        key: Tuple[Union[slice, int], Union[slice, int], Union[slice, int]]
            Zero-based (row, column, sample) indices into the total pixel
            matrix

        Returns
        -------
        numpy.ndarray
            Region of pixels spanning one or more tiles

        """
        if not isinstance(key, tuple) or len(key) != 3:
            raise ValueError('Encountered unexpected key.')

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
            raise TypeError('Row index must be an integer or a slice.')
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
        region_row_stop: Union[int, None]
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
            raise TypeError('Column index must be an integer or a slice.')
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
        region_col_stop: Union[int, None]
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
            raise TypeError('Sample index must be an integer or a slice.')

        frame_position_mapping = {}
        for (r, c) in itertools.product(row_tile_range, col_tile_range):
            tile_index = self.get_tile_index((r, c))
            frame_index = self._frame_indices[tile_index]
            # (Column, Row) position of the frame in the region pixel matrix.
            # The top left frame is located at (1, 1).
            # These tile positions will subsequently be passed to the
            # assemble_total_pixel_matrix() function, which expects the tile
            # positions to be provided according the specification of the DICOM
            # Column/Row Position In Total Image Pixel Matrix attributes.
            frame_position_mapping[frame_index] = (
                (
                    self._tile_positions[tile_index, 0] -
                    (row_start_tile_index * self._rows)
                ),
                (
                    self._tile_positions[tile_index, 1] -
                    (col_start_tile_index * self._cols)
                ),
            )

        frame_indices = [key for key in frame_position_mapping.keys()]
        frame_array_mapping = self._retrieve_and_decode_frames(frame_indices)
        tiles = [frame_array_mapping[i] for i in frame_indices]
        tile_positions: np.typing.NDArray[np.uint32] = np.array([
            frame_position_mapping[i] for i in frame_indices
        ])
        if tile_positions.shape[0] > 0:
            extended_region = assemble_total_pixel_matrix(
                tiles=tiles,
                tile_positions=tile_positions,
                total_pixel_matrix_columns=int(
                    np.max(tile_positions[:, 1]) + self._cols
                ),
                total_pixel_matrix_rows=int(
                    np.max(tile_positions[:, 0]) + self._rows
                )
            )

            # Sometimes, the stop index is off by one due to a rounding error.
            # If that's the case, let's adjust the (negative) stop index to
            # ensure that the resulting region is of the expected shape.
            if region_row_stop is not None:
                region_row_stop -= (
                    (
                        extended_region.shape[0] -
                        region_row_start + region_row_stop
                    ) -
                    (row_stop - row_start)
                )
            else:
                region_row_stop = -(
                    (
                        extended_region.shape[0] -
                        region_row_start
                    ) -
                    (row_stop - row_start)
                )
                if region_row_stop == 0:
                    region_row_stop = None
            if region_col_stop is not None:
                region_col_stop -= (
                    (
                        extended_region.shape[1] -
                        region_col_start + region_col_stop
                    ) -
                    (col_stop - col_start)
                )
            else:
                region_col_stop = -(
                    (
                        extended_region.shape[1] -
                        region_col_start
                    ) -
                    (col_stop - col_start)
                )
                if region_col_stop == 0:
                    region_col_stop = None

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
                if region_row_stop is not None:
                    if region_row_stop < 0:
                        adjusted_region_row_stop = np.sum([
                            self._metadata.TotalPixelMatrixRows,
                            region_row_stop
                        ])
                    else:
                        adjusted_region_row_stop = region_row_stop
                else:
                    adjusted_region_row_stop = region_row_stop
                region_row_diff = adjusted_region_row_stop - region_row_start
                shape.append(max([0, region_row_diff]))

            if len(col_tile_range) == 0:
                shape.append(0)
            else:
                if region_col_stop is not None:
                    if region_col_stop < 0:
                        adjusted_region_col_stop = np.sum([
                            self._metadata.TotalPixelMatrixColumns,
                            region_col_stop
                        ])
                    else:
                        adjusted_region_col_stop = region_col_stop
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

    def _read_tiles(
        self,
        key: Union[int, Sequence[int], slice]
    ) -> np.ndarray:
        """Read individual tiles.

        Retrieves one or more frames and decodes them.

        Parameters
        ----------
        key: Union[int, Sequence[int], slice]
            Zero-based tile indices into the flattened list of tiles in the
            total pixel matrix

        Returns
        -------
        numpy.ndarray
            A single 3D array or a sequence of 3D arrays, where each array
            represents a tile (rows x columns x samples)

        """
        indices = []
        if isinstance(key, int):
            indices.append(key)
        elif isinstance(key, Sequence):
            if isinstance(key[0], int):
                indices.extend([k for k in key])
            else:
                raise TypeError('Wrong tile index.')
        elif isinstance(key, slice):
            indices.extend(list(range(key.start, key.stop)))
        else:
            raise TypeError('Wrong tile index.')

        frame_indices = self._sorted_frame_indices[indices]
        tiles = self._retrieve_and_decode_frames(frame_indices)
        if len(indices) > 1:
            return np.stack([tiles[i] for i in frame_indices])
        elif len(indices) == 1:
            return [tiles[i] for i in frame_indices][0]
        else:
            raise IndexError(f'Could not find tiles: {indices}')

    def __getitem__(
        self,
        key: Union[
            Tuple[Union[slice, int], Union[slice, int], Union[slice, int]],
            int,
            Sequence[int],
            slice
        ]
    ) -> np.ndarray:
        error_message = (
            'Key must have type Tuple[Union[int, slice], ...], '
            'int, Sequence[int], or slice.'
        )
        if isinstance(key, tuple):
            if not isinstance(key[0], (int, slice)):
                raise TypeError(error_message)
            if len(key) != 3:
                raise ValueError(
                    'Index must be a three-dimensional to specify the '
                    'extent of the image region along the column direction '
                    '(top to bottom), the row direction (left to right), and '
                    'the sample direction (R, G, B in case of a color image).'
                )
            return self._read_region(key)  # type: ignore
        else:
            if not isinstance(key, (int, Sequence, slice)):
                raise TypeError(error_message)
            if isinstance(key, Sequence):
                if not isinstance(key[0], int):
                    raise TypeError(error_message)
            return self._read_tiles(key)


class TotalPixelMatrixSampler:

    """Class for sampling regions of a total pixel matrix.

    Regions are sampled from a regular 2D Cartesian grid, where each region has
    the same dimensions. Upon sampling, individual regions may optionally be
    padded at one or more borders using pixels from adjacent regions.
    Sampling can be constraint to a subset of the grid.

    """

    def __init__(
        self,
        matrix: TotalPixelMatrix,
        region_dimensions: Tuple[int, int],
        bounding_box: Optional[
            Tuple[Tuple[int, int], Tuple[int, int]]
        ] = None,
        tile_grid_positions: Optional[
            Union[Sequence[Tuple[int, int]], np.ndarray]
        ] = None,
        padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]] = 0,
    ):
        """

        Parameters
        ----------
        matrix: dicomslide.TotalPixelMatrix
            Total pixel matrix
        region_dimensions: Tuple[int, int]
            Height (rows) and width (columns) of sampled regions
        bounding_box: Union[Tuple[Tuple[int, int], Tuple[int, int]], None], optional
            Bounding box of region of interest within total pixel matrix from
            which regions should be sampled
        tile_grid_positions: Union[Sequence[Tuple[int, int]], numpy.ndarray, None], optional
            Grid position of tiles that intersect with the region of interest
            within the total pixel matrix from which regions should be sampled.
            Each grid position is a zero-based (row, column) index into the
            tile grid of the total pixel matrix.
        padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]], optional
            Padding on each border of the sampled region using pixels from
            neighboring regions. If a single integer is provided, the value is
            used to pad all four borders with the same number of pixels. If a
            sequence of length 2 is provided, the two values are used to pad
            the left/right and top/bottom border, respectively. If a sequence
            of length 4 is provided, the four values are used to pad the left,
            top, right, and bottom borders respectively.

        Note
        ----
        If `bounding_box` and `tile_grid_positions` are provided,
        `tile_grid_positions` are ignored.

        """  # noqa: E501
        self._matrix = matrix
        self._padding: Tuple[int, int, int, int]
        if isinstance(padding, int):
            self._padding = (padding, padding, padding, padding)
        elif isinstance(padding, tuple):
            if len(padding) == 2:
                self._padding = (padding[0], padding[1], padding[0], padding[1])
            elif len(padding) == 4:
                self._padding = (
                    padding[0],
                    padding[1],
                    padding[2],  # type: ignore
                    padding[3],  # type: ignore
                )
            else:
                raise ValueError(
                    'If argument "padding" is a tuple, its length must be '
                    'either 2 or 4.'
                )
        else:
            raise TypeError(
                'Argument "padding" must be either an integer or a tuple.'
            )
        self._region_shape = (
            region_dimensions[0],
            region_dimensions[1],
            matrix.tile_shape[2]
        )
        self._region_grid_shape = (
            int(np.ceil(matrix.shape[0] / self._region_shape[0])),
            int(np.ceil(matrix.shape[1] / self._region_shape[1])),
        )
        self._region_grid_coordinates = np.array([
            (r, c)
            for r, c in itertools.product(
                range(self._region_grid_shape[0]),
                range(self._region_grid_shape[1]),
            )
        ])

        if bounding_box is not None:
            if not isinstance(bounding_box, Sequence):
                raise TypeError('Argument "bounding_box" must be a sequence.')
            if len(bounding_box) != 2:
                raise ValueError(
                    'Argument "bounding_box" must be a sequence of length 2.'
                )
            if tile_grid_positions is not None:
                logger.warning(
                    'arguments "bounding_box" and "tile_grid_positions" were '
                    'both provided and "tile_positions" will be ignored'
                )
            offset, size = bounding_box
            if not isinstance(offset, Sequence):
                raise TypeError(
                    'First item of argument "bounding_box" must be a sequence.'
                )
            if len(offset) != 2:
                raise ValueError(
                    'First item of argument "bounding_box" must be a sequence '
                    'of length 2.'
                )
            if not isinstance(size, Sequence):
                raise TypeError(
                    'Second item of argument "bounding_box" must be a sequence.'
                )
            if len(size) != 2:
                raise ValueError(
                    'Second item of argument "bounding_box" must be a sequence '
                    'of length 2.'
                )

            if (
                (offset[0] + size[0]) > matrix.shape[0] or
                (offset[1] + size[1]) > matrix.shape[1]
            ):
                raise ValueError(
                    'Bounding box must not extend beyond total pixel matrix.'
                )
            box_start_grid_coordinates = (
                int(np.floor(offset[0] / self._region_shape[0])),
                int(np.floor(offset[1] / self._region_shape[1])),
            )
            box_end_grid_coordinates = (
                int(np.ceil((offset[0] + size[0]) / self._region_shape[0])),
                int(np.ceil((offset[1] + size[1]) / self._region_shape[1])),
            )
            self._selected_region_grid_coordinates = np.array([
                (r, c)
                for r, c in itertools.product(
                    range(
                        box_start_grid_coordinates[0],
                        box_end_grid_coordinates[0],
                    ),
                    range(
                        box_start_grid_coordinates[1],
                        box_end_grid_coordinates[1],
                    )
                )
            ])

        elif tile_grid_positions is not None:
            grid_coordinates = []
            for r, c in tile_grid_positions:
                tile_index = matrix.get_tile_index((r, c))
                offset = matrix.get_tile_position(tile_index)
                size = matrix.tile_shape[:2]
                start_grid_coordinates = (
                    int(np.floor(offset[0] / self._region_shape[0])),
                    int(np.floor(offset[1] / self._region_shape[1])),
                )
                grid_coordinates.append(start_grid_coordinates)
                stop_grid_coordinates = (
                    int(np.ceil((offset[0] + size[0]) / self._region_shape[0])),
                    int(np.ceil((offset[1] + size[1]) / self._region_shape[1])),
                )
                grid_coordinates.append(stop_grid_coordinates)
            grid_coordinates = np.array(grid_coordinates)
            self._selected_region_grid_coordinates = np.unique(
                grid_coordinates,
                axis=0
            )

        else:
            self._selected_region_grid_coordinates = np.array(
                self._region_grid_coordinates
            )

        self._current_index = 0

    def __len__(self):
        return self._selected_region_grid_coordinates.shape[0]

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self):
        if self._current_index >= len(self):
            raise StopIteration
        index = int(self._current_index)
        self._current_index += 1
        return self[index]

    @property
    def matrix(self) -> TotalPixelMatrix:
        """dicomslide.TotalPixelMatrix: Total pixel matrix"""
        return self._matrix

    @property
    def padding(self) -> Tuple[int, int, int, int]:
        """Tuple[int, int, int, int]: Padding at the left, top, right, and
        bottom of each sampled region

        """
        return self._padding

    @property
    def region_shape(self) -> Tuple[int, int, int]:
        """Tuple[int, int, int]: Number of pixel rows, pixel columns, and
        samples per pixel of a region

        """
        return self._region_shape

    @property
    def padded_region_shape(self) -> Tuple[int, int, int]:
        """Tuple[int, int, int]: Number of pixel rows, pixel columns, and
        samples per pixel of sampled region with overlapping pixels from
        neighboring regions

        """
        left, top, right, bottom = self.padding
        return (
            self.region_shape[0] + top + bottom,
            self.region_shape[1] + left + right,
            self.matrix.tile_shape[2]
        )

    def get_region_grid_position(self, index: int) -> Tuple[int, int]:
        """Get position of sampled region in the grid.

        Parameters
        ----------
        index: int
            Zero-based index of the sampled region

        Returns
        -------
        Tuple[int, int]
            Zero-based (row, column) grid position

        """
        r, c = self._selected_region_grid_coordinates[index, :]
        return (int(r), int(c))

    def __getitem__(self, index: int) -> np.ndarray:
        """Get region.

        The region includes the pixels of the tile and potentially parts of
        additional frames to the left, top, right, and bottom of the frame. How
        many pixels of neighboring tiles are included in the region depends on
        the value of the `padding` attribute.

        Parameters
        ----------
        index: int
            Zero-based index of the sampled region

        Returns
        -------
        numpy.ndarray
            Image region including pixels of the tile and potentially
            parts of neighboring tiles

        """
        region_rows, region_cols, _ = self.region_shape
        padded_region_rows, padded_region_cols, _ = self.padded_region_shape
        padded_region = np.zeros(
            self.padded_region_shape,
            dtype=self.matrix.dtype
        )

        r, c = self.get_region_grid_position(index)
        left, top, right, bottom = self.padding
        total_row_start = r * region_rows
        total_col_start = c * region_cols
        total_rows, total_cols, _ = self.matrix.shape

        if total_row_start == 0:
            padded_region_row_start = top
            adjusted_total_row_start = total_row_start
            adjusted_total_row_end = (
                adjusted_total_row_start + padded_region_rows - top
            )
        else:
            padded_region_row_start = 0
            adjusted_total_row_start = total_row_start - top
            adjusted_total_row_end = (
                adjusted_total_row_start + padded_region_rows
            )
        row_diff = adjusted_total_row_end - total_rows
        if row_diff > 0:
            padded_region_row_end = -row_diff
            adjusted_total_row_end -= row_diff
        else:
            padded_region_row_end = None

        if total_col_start == 0:
            padded_region_col_start = left
            adjusted_total_col_start = total_col_start
            adjusted_total_col_end = (
                adjusted_total_col_start + padded_region_cols - left
            )
        else:
            padded_region_col_start = 0
            adjusted_total_col_start = total_col_start - left
            adjusted_total_col_end = (
                adjusted_total_col_start + padded_region_cols
            )
        col_diff = adjusted_total_col_end - total_cols
        if col_diff > 0:
            padded_region_col_end = -col_diff
            adjusted_total_col_end -= col_diff
        else:
            padded_region_col_end = None

        padded_region[
            padded_region_row_start:padded_region_row_end,
            padded_region_col_start:padded_region_col_end,
            :
        ] = self.matrix[
            adjusted_total_row_start:adjusted_total_row_end,
            adjusted_total_col_start:adjusted_total_col_end,
            :
        ]

        return padded_region
