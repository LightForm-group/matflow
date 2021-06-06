"""`matflow.cli.py`

Module that exposes a command line interface for `matflow`.

"""
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
@click.argument('workflow_path', type=click.Path(exists=True))
def go(workflow_path, directory=None):
    """Generate and submit a new Workflow."""
    print('matflow.cli.go', flush=True)
    api.submit_workflow(workflow_path, directory=directory)


@cli.command()
@click.option('--task-idx', '-t', type=click.INT, required=True)
@click.option('--iteration-idx', '-i', type=click.INT, required=True)
@click.option('--directory', '-d', type=click.Path(exists=True))
@click.option('--array', is_flag=True)
def prepare_task(task_idx, iteration_idx, directory=None, array=False):
    print('matflow.cli.prepare_task', flush=True)
    api.prepare_task(task_idx, iteration_idx, directory, is_array=array)


@cli.command()
@click.option('--task-idx', '-t', type=click.INT, required=True)
@click.option('--element-idx', '-e', type=click.INT, required=True)
@click.option('--directory', '-d', type=click.Path(exists=True))
@click.option('--array', is_flag=True)
def prepare_task_element(task_idx, element_idx, directory=None, array=False):
    print('matflow.cli.prepare_task_element', flush=True)
    api.prepare_task_element(task_idx, element_idx, directory, is_array=array)


@cli.command()
@click.option('--task-idx', '-t', type=click.INT, required=True)
@click.option('--iteration-idx', '-i', type=click.INT, required=True)
@click.option('--directory', '-d', type=click.Path(exists=True))
@click.option('--array', is_flag=True)
def process_task(task_idx, iteration_idx, directory=None, array=False):
    print('matflow.cli.process_task', flush=True)
    api.process_task(task_idx, iteration_idx, directory, is_array=array)


@cli.command()
@click.option('--task-idx', '-t', type=click.INT, required=True)
@click.option('--element-idx', '-e', type=click.INT, required=True)
@click.option('--directory', '-d', type=click.Path(exists=True))
@click.option('--array', is_flag=True)
def process_task_element(task_idx, element_idx, directory=None, array=False):
    print('matflow.cli.process_task_element', flush=True)
    api.process_task_element(task_idx, element_idx, directory, is_array=array)


@cli.command()
@click.option('--task-idx', '-t', type=click.INT, required=True)
@click.option('--element-idx', '-e', type=click.INT, required=True)
@click.option('--directory', '-d', type=click.Path(exists=True))
def run_python_task(task_idx, element_idx, directory=None):
    print('matflow.cli.run_python_task', flush=True)
    api.run_python_task(task_idx, element_idx, directory)


@cli.command()
@click.option('--task-idx', '-t', type=click.INT, required=True)
@click.option('--iteration-idx', '-i', type=click.INT, required=True)
@click.option('--directory', '-d', type=click.Path(exists=True))
def prepare_sources(task_idx, iteration_idx, directory=None):
    print('matflow.cli.prepare_sources', flush=True)
    api.prepare_sources(task_idx, iteration_idx, directory)


@cli.command()
@click.argument('schema_source_path', type=click.Path(exists=True))
def append_schema_source(schema_source_path):
    api.append_schema_source(schema_source_path)


@cli.command()
@click.argument('schema_source_path', type=click.Path(exists=True))
def prepend_schema_source(schema_source_path):
    api.prepend_schema_source(schema_source_path)


@cli.command()
def validate():
    """Load and validate task schemas against available extensions."""
    api.validate()


@cli.command()
@click.option('--provider', '-p', required=True)
def cloud_connect(provider):
    api.cloud_connect(provider)


@cli.command()
@click.argument('directory', type=click.Path(exists=True))
def kill(directory):
    """Kill all pending and executing tasks."""
    api.kill(directory)


@cli.command()
@click.option('--iteration-idx', '-i', type=click.INT, required=True)
@click.option('--directory', '-d', type=click.Path(exists=True))
def write_element_directories(iteration_idx, directory=None):
    api.write_element_directories(iteration_idx, directory)


@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.argument('archive')
def archive(directory, archive):
    api.archive(directory, archive)


if __name__ == '__main__':
    cli()
