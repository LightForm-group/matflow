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
@click.option('--directory', '-d')
@click.argument('profile', type=click.Path(exists=True))
def go(profile, directory=None):
    """Generate and submit a new Workflow."""
    print('matflow.cli.go', flush=True)
    api.go(profile_path=profile, directory=directory)


@cli.command()
@click.option('--task-idx', '-t', type=click.INT, required=True)
@click.option('--directory', '-d', type=click.Path(exists=True))
def prepare_task(task_idx, directory=None):
    print('matflow.cli.prepare_task', flush=True)
    api.prepare_task(task_idx, directory)


@cli.command()
@click.option('--task-idx', '-t', type=click.INT, required=True)
@click.option('--directory', '-d', type=click.Path(exists=True))
def process_task(task_idx, directory=None):
    print('matflow.cli.process_task', flush=True)
    api.process_task(task_idx, directory)


if __name__ == '__main__':
    cli()
