"""Dummy DICOM data sets for testing purposes."""
import itertools
from datetime import datetime
from typing import List, Sequence, Tuple, Union

import numpy as np
import highdicom as hd
from PIL.ImageCms import ImageCmsProfile, createProfile
from pydicom.dataset import Dataset
from pydicom.encaps import encapsulate
from pydicom.tag import TupleTag
from pydicom.uid import (
    JPEGBaseline8Bit,
    JPEG2000Lossless,
    ImplicitVRLittleEndian,
    ExplicitVRLittleEndian,
)


class VLWholeSlideMicroscopyImage(hd.SOPClass):
    """Dummy DICOM VL Whole Slide Microscopy Image instance.

    Pixel Data element is populated with small frames for testing purposes.
    """

    def __init__(
        self,
        study_instance_uid: str,
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        image_type: Tuple[str, str, str, str],
        rows: int,
        columns: int,
        total_pixel_matrix_rows: int,
        total_pixel_matrix_columns: int,
        samples_per_pixel: int,
        number_of_focal_planes: int,
        number_of_optical_paths: int,
        optical_path_identifiers: Sequence[int],
        extended_depth_of_field: bool,
        pixel_spacing: Tuple[float, float],
        image_position: Tuple[float, float, float],
        image_orientation: Tuple[float, float, float, float, float, float],
        dimension_organization_type: Union[
            hd.DimensionOrganizationTypeValues, str
        ],
        transfer_syntax_uid: str,
        frame_of_reference_uid: str,
        container_id: str,
        specimen_id: str,
        specimen_uid: str,
        spacing_between_slices: float = 0.001,
    ) -> None:
        """Construct object.

        Parameters
        ----------
        study_instance_uid: str
            Unique study identifier
        series_instance_uid: str
            Unique series identifier
        sop_instance_uid: str
            Unique instance identifier
        series_number: int
            Number of the series in the study
        instance_number: int
            Number of the instance in the series
        image_type: Tuple[str, str, str, str]
            Image type (e.g., ``('ORIGINAL', 'PRIMARY', 'VOLUME', 'NONE')``)
        rows: int
            Number of rows in each image tile (frame height)
        columns: int
            Number of columns in each image tile (frame width)
        total_pixel_matrix_rows: int
            Number of rows in the image (total height)
        total_pixel_matrix_columns: int
            Number of column in the image (total width)
        samples_per_pixel: int
            Number of colors per pixel
        number_of_focal_planes: int
            Number of focal planes
        number_of_optical_paths: int
            Number of optical paths (channels)
        extended_depth_of_field: bool
            Whether image pixels where created by combining multiple focal
            planes (z-stacking)
        pixel_spacing: Tuple[float, float]
            Size of each pixel along the column and row axis of the image in mm
        image_position: Tuple[float, float, float]
            Offset of the top left pixel of the image along the X, Y, and Z
            axis of the three dimensional slide coordinate system in mm (X and
            Y axes) and um (Z axis).
        image_orientation: Tuple[float, float, float, float, float, float]
            Direction cosines of the first row and column of the image in the
            three dimensional slide coordinate system. The first three values
            are the direction cosines for the first row (left -> right) with
            respect to the X axis of the slide (shorter side of the slide).
            They express the direction change in (x, y, z) coordinates along
            each image row, i.e. increasing column pixel position. The second
            three values are the direction cosines for the first column
            (top -> bottom) with respect to the Y axis of the slide (longer
            side of the slide). They express the direction change in (x, y, z)
            coordinates along each image column, i.e. increasing row pixel
            position.
        dimension_organization_type: Union[highdicom.DimensionOrganizationTypeValues, str]
            The way image frames of the tiled total pixel matrix are organized
            (if ``TILED_SPARSE`` the positional metadata for each frame is
            explicitly specified in form an item of the Per-Frame Functional
            Groups Sequence attribute and can be looked up, if ``TILED_FULL``
            this attribute is omitted and the position of each frame is
            implicitly specified and needs to be computed).
        transfer_syntax_uid: str
            Transfer syntax for encoding of frames (supported are JPEG Baseline
            for 8-bit images and JPEG 2000 for 16-bit images)
        frame_of_reference_uid: str
            Unique identifier of the frame of reference (slide coordinate
            system)
        container_id: str
            Identifier of the container (glass slide)
        specimen_id: str
            Identifier of the specimen (tissue section)
        specimen_uid: str
            Unique identifier of the specimen (tissue section)

        Returns
        -------
        pydicom.dataset.Dataset
            VL Whole Slide Microscopy Image instance

        Warning
        -------
        This is not implemented in a memory efficient manner, consider creating
        small images for testing purposes.

        """  # noqa
        super().__init__(
            study_instance_uid=study_instance_uid,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid="1.2.840.10008.5.1.4.1.1.77.1.6",
            instance_number=instance_number,
            manufacturer="Tester",
            modality="SM",
            transfer_syntax_uid=transfer_syntax_uid,
        )

        self.ManufacturerModelName = "Mock"
        self.DeviceSerialNumber = "X.X.X"
        self.SoftwareVersions = "0.0"

        image_flavor = image_type[2]
        if image_flavor in {'THUMBNAIL', 'LABEL', 'OVERVIEW'}:
            if (
                rows < total_pixel_matrix_rows or
                columns < total_pixel_matrix_columns
            ):
                raise ValueError(
                    'VL Whole Slide Microscopy Image with flavor '
                    f'"{image_flavor}" must contain only a single tile, '
                    'i.e., only one frame per optical path and focal plane.'
                )
        self.ImageType = image_type

        self.SamplesPerPixel = samples_per_pixel
        if samples_per_pixel == 3:
            if transfer_syntax_uid in (JPEGBaseline8Bit, JPEG2000Lossless):
                self.PhotometricInterpretation = "YBR_FULL_422"
            else:
                self.PhotometricInterpretation = "RGB"
            self.BitsAllocated = 8
            pixel_value = 255
            self.PlanarConfiguration = 0
            image_shape = [rows, columns, samples_per_pixel]
            image_dtype = np.uint8
        elif samples_per_pixel == 1:
            self.PhotometricInterpretation = "MONOCHROME2"
            self.BitsAllocated = 16
            pixel_value = 0
            self.RescaleIntercept = 0
            self.RescaleSlope = 1
            self.PresentationLUTShape = "IDENTITY"
            image_shape = [rows, columns]
        else:
            raise ValueError("Unsupported number of samples per pixel.")
        image_dtype = np.dtype(f'uint{self.BitsAllocated}')
        self.BitsStored = self.BitsAllocated
        self.HighBit = self.BitsAllocated - 1
        self.PixelRepresentation = 0
        self.OpticalPathSequence = []
        wavelengths = (488.0, 555.0, 647.0)
        if number_of_optical_paths > len(wavelengths):
            raise ValueError("Unsupported number of optical paths.")
        if (
            number_of_optical_paths > 1 and
            samples_per_pixel == 3
        ):
            raise ValueError(
                "Unsupported number of optical paths for "
                "specified photometric interpretation."
            )
        for i, optical_path_id in enumerate(optical_path_identifiers):
            optical_path_item = Dataset()
            optical_path_item.OpticalPathIdentifier = str(optical_path_id)
            if self.PhotometricInterpretation == "MONOCHROME2":
                optical_path_item.IlluminationWaveLength = wavelengths[i]
                optical_path_item.IlluminationTypeCodeSequence = [
                    hd.sr.CodedConcept(
                        value="111741",
                        meaning="Transmission illumination",
                        scheme_designator="DCM",
                    ),
                ]
            else:
                optical_path_item.IlluminationColorCodeSequence = [
                    hd.sr.CodedConcept(
                        value="414298005",
                        meaning="Full spectrum",
                        scheme_designator="SCT",
                    ),
                ]
                optical_path_item.IlluminationTypeCodeSequence = [
                    hd.sr.CodedConcept(
                        value="111744",
                        meaning="Brightfield illumination",
                        scheme_designator="DCM",
                    ),
                ]
                icc_profile = ImageCmsProfile(createProfile("sRGB"))
                optical_path_item.ICCProfile = icc_profile.tobytes()
            self.OpticalPathSequence.append(optical_path_item)

        self.Rows = rows
        self.Columns = columns
        self.TotalPixelMatrixColumns = total_pixel_matrix_columns
        self.TotalPixelMatrixRows = total_pixel_matrix_rows
        self.ImagedVolumeWidth = round(
            total_pixel_matrix_columns * pixel_spacing[1], 6
        )
        self.ImagedVolumeHeight = round(
            total_pixel_matrix_rows * pixel_spacing[0], 6
        )
        slice_thickness = 0.0001
        self.ImagedVolumeDepth = (
            (number_of_focal_planes * spacing_between_slices) +
            slice_thickness
        )
        row_direction_cosines = np.array(image_orientation[:3])
        col_direction_cosines = np.array(image_orientation[3:])
        if not (
            np.dot(row_direction_cosines, col_direction_cosines) == 0 and
            np.dot(row_direction_cosines, row_direction_cosines) == 1 and
            np.dot(col_direction_cosines, col_direction_cosines) == 1
        ):
            raise ValueError("Incorrect image orientation.")
        self.ImageOrientationSlide = list(image_orientation)
        tpmo_item = Dataset()
        tpmo_item.XOffsetInSlideCoordinateSystem = image_position[0]
        tpmo_item.YOffsetInSlideCoordinateSystem = image_position[1]
        self.TotalPixelMatrixOriginSequence = [tpmo_item]
        tiles_per_column = int(np.ceil(total_pixel_matrix_rows / rows))
        tiles_per_row = int(np.ceil(total_pixel_matrix_columns / columns))
        if extended_depth_of_field:
            self.ExtendedDepthOfField = "YES"
            self.NumberOfFocalPlanes = 10
            self.DistanceBetweenFocalPlanes = 2  # arbitrary value
        else:
            self.ExtendedDepthOfField = "NO"
        self.NumberOfFrames = int(
            np.product([
                number_of_optical_paths,
                number_of_focal_planes,
                tiles_per_column,
                tiles_per_row,
            ])
        )

        dimension_organization_uid = hd.UID()
        self.FrameOfReferenceUID = frame_of_reference_uid
        self.PositionReferenceIndicator = "SLIDE_CORNER"
        dim_org_item = Dataset()
        dim_org_item.DimensionOrganizationUID = dimension_organization_uid
        self.DimensionOrganizationSequence = [dim_org_item]

        self.ContainerIdentifier = container_id
        self.IssuerOfTheContainerIdentifierSequence = None
        self.ContainerTypeCodeSequence = [
            hd.sr.CodedConcept(
                value="433466003",
                meaning="Microscope Slide",
                scheme_designator="SCT",
            ),
        ]
        specimen_description_item = hd.SpecimenDescription(
            specimen_id=specimen_id,
            specimen_uid=specimen_uid
        )
        self.SpecimenDescriptionSequence = [specimen_description_item]

        self.ContentDate = datetime.now().strftime("%Y%m%d")
        self.ContentTime = datetime.now().strftime("%H%M%S")
        self.AcquisitionDateTime = datetime.now().strftime("%Y%m%d%H%M%S")
        self.AcquisitionContextSequence: List[Dataset] = []
        self.VolumetricProperties = "VOLUME"
        self.SpecimenLabelInImage = "NO"
        self.BurnedInAnnotation = "NO"
        self.FocusMethod = "AUTO"

        sfgs_item = Dataset()
        pms_item = Dataset()
        pms_item.PixelSpacing = [round(space, 6) for space in pixel_spacing]
        if number_of_focal_planes > 1 and not extended_depth_of_field:
            pms_item.SliceThickness = slice_thickness
            pms_item.SpacingBetweenSlices = spacing_between_slices
        else:
            pms_item.SliceThickness = slice_thickness
        sfgs_item.PixelMeasuresSequence = [pms_item]
        wsmifts_item = Dataset()
        wsmifts_item.FrameType = self.ImageType
        sfgs_item.WholeSlideMicroscopyImageFrameTypeSequence = [wsmifts_item]
        self.SharedFunctionalGroupsSequence = [sfgs_item]

        dimension_organization_type = hd.DimensionOrganizationTypeValues(
            dimension_organization_type
        )
        self.DimensionOrganizationType = dimension_organization_type.value
        if dimension_organization_type.value == "TILED_SPARSE":
            self.DimensionIndexSequence = []
            x_item = Dataset()
            x_item.FunctionalGroupPointer = TupleTag((0x0048, 0x021A))
            x_item.DimensionIndexPointer = TupleTag((0x0040, 0x021E))
            x_item.DimensionDescriptionLabel = "Column position index"
            x_item.DimensionOrganizationUID = dimension_organization_uid
            y_item = Dataset()
            y_item.FunctionalGroupPointer = TupleTag((0x0048, 0x021A))
            y_item.DimensionIndexPointer = TupleTag((0x0040, 0x021F))
            y_item.DimensionDescriptionLabel = "Row position index"
            y_item.DimensionOrganizationUID = dimension_organization_uid
            z_item = Dataset()
            z_item.FunctionalGroupPointer = TupleTag((0x0048, 0x021A))
            z_item.DimensionIndexPointer = TupleTag((0x0040, 0x074A))
            z_item.DimensionDescriptionLabel = "Z axis index"
            z_item.DimensionOrganizationUID = dimension_organization_uid
            c_item = Dataset()
            c_item.FunctionalGroupPointer = TupleTag((0x0048, 0x0207))
            c_item.DimensionIndexPointer = TupleTag((0x0048, 0x0106))
            c_item.DimensionDescriptionLabel = "Channel index"
            c_item.DimensionOrganizationUID = dimension_organization_uid
            self.DimensionIndexSequence = [
                x_item,
                y_item,
                z_item,
                c_item,
            ]
            self.PerFrameFunctionalGroupsSequence = []
        else:
            self.TotalPixelMatrixFocalPlanes = number_of_focal_planes
            self.NumberOfOpticalPaths = number_of_optical_paths

        frames = []
        # Generate frames in the order defined by TILED_FULL organization type.
        frame_iterator = itertools.product(
            range(number_of_optical_paths),
            range(number_of_focal_planes),
            range(tiles_per_column),
            range(tiles_per_row),
        )
        for c_index, z_index, tile_row_index, tile_col_index in frame_iterator:
            if dimension_organization_type.value == "TILED_SPARSE":

                col_index = tile_col_index * self.Columns
                row_index = tile_row_index * self.Rows

                # See this article for further information about the algorithm:
                # https://nipy.org/nibabel/dicom/dicom_orientation.html
                img_pos = np.array(
                    [
                        tpmo_item.XOffsetInSlideCoordinateSystem,
                        tpmo_item.YOffsetInSlideCoordinateSystem,
                        0,
                    ]
                )

                mapping = np.concatenate(
                    [
                        row_direction_cosines[..., None].T * pixel_spacing[0],
                        col_direction_cosines[..., None].T * pixel_spacing[1],
                        np.zeros((3, 1)).T,
                        img_pos[..., None].T,
                    ]
                ).T
                mapping = np.concatenate([mapping, np.array([[0, 0, 0, 1]])])
                index = np.array([[col_index, row_index, 0, 1]]).T
                coordinates = np.dot(mapping, index)
                x_offset = coordinates[0][0]
                y_offset = coordinates[1][0]
                if number_of_focal_planes > 1 and not extended_depth_of_field:
                    z_offset = z_index * spacing_between_slices
                else:
                    z_offset = 0.0

                pp_item = Dataset()
                pp_item.XOffsetInSlideCoordinateSystem = x_offset
                pp_item.YOffsetInSlideCoordinateSystem = y_offset
                pp_item.ZOffsetInSlideCoordinateSystem = z_offset
                pp_item.ColumnPositionInTotalImagePixelMatrix = col_index + 1
                pp_item.RowPositionInTotalImagePixelMatrix = row_index + 1

                frame_content_item = Dataset()
                frame_content_item.DimensionIndexValues = [
                    row_index + 1,
                    col_index + 1,
                    z_index + 1,
                    c_index + 1,
                ]

                path_id_item = Dataset()
                path_id_item.OpticalPathIdentifier = str(
                    optical_path_identifiers[c_index]
                )

                pffg_item = Dataset()
                pffg_item.FrameContentSequence = [frame_content_item]
                pffg_item.OpticalPathIdentificationSequence = [path_id_item]
                pffg_item.PlanePositionSlideSequence = [pp_item]
                self.PerFrameFunctionalGroupsSequence.append(pffg_item)

            tile = np.ones(image_shape, dtype=image_dtype)
            tile *= image_dtype.type(pixel_value)
            if transfer_syntax_uid == JPEGBaseline8Bit:
                self.LossyImageCompression = "01"
                self.LossyImageCompressionMethod = "ISO_10918_1"
                self.LossyImageCompressionRatio = 10
            elif transfer_syntax_uid == JPEG2000Lossless:
                self.LossyImageCompression = "00"
            elif transfer_syntax_uid in (
                ImplicitVRLittleEndian,
                ExplicitVRLittleEndian,
            ):
                self.LossyImageCompression = "00"
            else:
                raise ValueError(
                    f'Unsupported transfer syntax "{transfer_syntax_uid}".'
                )

            encoded_frame = hd.frame.encode_frame(
                array=tile,
                transfer_syntax_uid=transfer_syntax_uid,
                bits_allocated=self.BitsAllocated,
                bits_stored=self.BitsStored,
                photometric_interpretation=self.PhotometricInterpretation,
                pixel_representation=self.PixelRepresentation,
                planar_configuration=getattr(
                    self,
                    'PlanarConfiguration',
                    None
                )
            )
            frames.append(encoded_frame)

        if self.file_meta.TransferSyntaxUID.is_encapsulated:
            self.PixelData = encapsulate(frames, has_bot=True)
        else:
            self.PixelData = b"".join(frames)
