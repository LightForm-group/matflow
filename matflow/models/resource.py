"""`matflow.models.jsonable.py`"""

from pathlib import Path


class Resource(object):

    def __init__(self, name, machine, base_path):

        self.name = name
        self.machine = machine
        self._base_path = str(base_path)

    def __repr__(self):
        out = (
            f'{self.__class__.__name__}('
            f'name={self.name!r}, '
            f'machine={self.machine!r}, '
            f'base_path={self._base_path!r}'
            f')'
        )
        return out

    @property
    def cloud_machine(self):
        if self.machine.is_dropbox:
            return self.machine
        else:
            raise ValueError(f'No cloud machine for resource "{self.name}".')

    @property
    def non_cloud_machines(self):
        if self.machine.is_dropbox:
            if self.machine.sync_client_paths:
                return self.machine.sync_client_paths
            else:
                return []
        else:
            raise ValueError(f'No non-cloud machine for resource "{self.name}".')

    @property
    def base_path(self):
        return Path(self._base_path)

    def to_jsonable(self):
        'Represent in a hickle-friendly format.'
        out = {
            'name': self.name,
            'machine': self.machine.name,
            'base_path': self._base_path,
        }
        return out


class ResourceConnection(object):

    def __init__(self, source, destination, hostname=None):

        self.source = source
        self.destination = destination
        self.hostname = hostname

    def to_jsonable(self):
        'Represent in a hickle-friendly format.'

        out = {
            'source': self.source.name,
            'destination': self.destination.name,
            'hostname': self.hostname,
        }
        return out


class Machine(object):

    def __init__(self, name, os_type, is_dropbox=False, sync_client_paths=None):

        self.name = name
        self.os_type = os_type
        self.is_dropbox = is_dropbox
        self.sync_client_paths = sync_client_paths

    def __repr__(self):
        out = (
            f'{self.__class__.__name__}('
            f'name={self.name!r}, '
            f'os_type={self.os_type!r}, '
            f'is_dropbox={self.is_dropbox!r}, '
            f'sync_client_paths={self.sync_client_paths!r}'
            f')'
        )
        return out

    def to_jsonable(self):
        'Represent in a hickle-friendly format.'

        sync_client_paths = [
            {
                'machine': i['machine'].name,
                'sync_path': i['sync_path'],
            }
            for i in self.sync_client_paths
        ]
        out = {
            'name': self.name,
            'os_type': self.os_type,
            'is_dropbox': self.is_dropbox,
            'sync_client_paths': sync_client_paths,
        }
        return out
