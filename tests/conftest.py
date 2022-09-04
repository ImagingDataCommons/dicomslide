import pytest
from dicomweb_client import DICOMfileClient


@pytest.fixture
def client(tmpdir):
    '''Instance of `dicomweb_client.api.DICOMwebClient`.'''
    url = f'file://{tmpdir}'
    return DICOMfileClient(url, recreate_db=True, in_memory=True)
