"""`matflow.cli.py`

Module that exposes a command line interface for `matflow`.

"""

from pathlib import Path

import click

from matflow import __version__
from matflow import api


@click.group()
@click.version_option(version=__version__)
def cli():
    pass


@cli.command()
@click.option('--directory', '-d')
@click.argument('profile', type=click.Path(exists=True))
def make(profile, directory=None):
    """Generate a new Workflow."""
    print('matflow.cli.make', flush=True)
    api.make_workflow(profile_path=profile, directory=directory)


@cli.command()
@click.argument('directory', type=click.Path(exists=True))
def proceed(directory):
    'Start/continue a workflow.'
    print('matflow.cli.proceed')
    api.proceed(directory)


if __name__ == '__main__':
    cli()
