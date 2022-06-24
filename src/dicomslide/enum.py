from enum import Enum


class ImageFlavors(Enum):

    """Enumerated values for image flavors."""

    VOLUME = 'VOLUME'
    LABEL = 'LABEL'
    OVERVIEW = 'OVERVIEW'
    THUMBNAIL = 'THUMBNAIL'


class ChannelTypes(Enum):

    """Enumerated values for channel types."""

    OPTICAL_PATH = 'OPTICAL_PATH'
    SEGMENT = 'SEGMENT'
    PARAMETER = 'PARAMETER'
