import pytest
from dicomweb_client import DICOMfileClient


@pytest.fixture
def client(tmpdir):
    '''Instance of `dicomweb_client.api.DICOMwebClient`.'''
    return DICOMfileClient(tmpdir, recreate_db=True)
