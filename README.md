[![DOI](https://zenodo.org/badge/219949875.svg)](https://zenodo.org/badge/latestdoi/219949875) [![PyPI version](https://badge.fury.io/py/matflow.svg)](https://badge.fury.io/py/matflow)

# MatFlow

MatFlow is a framework for running reproducible workflows in materials science developed in the EPSRC programme grant [LightForm](http://lightform.org.uk), a research programme on light alloy formability. It is a python program that interacts with software (open-source and proprietary) used in materials science via extensions (see supported extensions below). It is particularly suited for hybrid workflows
(involving experimental data and computational work), like for example HPC model calibration. Outputs,together with details of the worflow are automatically stored in an open source file format for post-processing, which Matflow can automatically upload to data repositories like [Zenodo](https://zenodo.org/).

See [this repository](https://github.com/LightForm-group/UoM-CSF-matflow) for information regarding a MatFlow installation.

## Extensions

MatFlow uses extension packages to interact with arbitrary software. Here is a list of current MatFlow extensions.

### Released/in-progress extensions
| Software | Description | Status | Version |
| ------ | ------------- | ------- | ------- |
| [DAMASK](https://damask.mpie.de/) | Düsseldorf Advanced Material Simulation Kit (crystal plasticity) | [Released](https://github.com/LightForm-group/matflow-damask) | [![PyPI version](https://img.shields.io/pypi/v/matflow-damask)](https://pypi.org/project/matflow-damask) |
| [MTEX](https://mtex-toolbox.github.io/) | Matlab toolbox for analyzing and modeling crystallographic textures | [Released](https://github.com/LightForm-group/matflow-mtex) | [![PyPI version](https://img.shields.io/pypi/v/matflow-mtex)](https://pypi.org/project/matflow-mtex) |
| [formable](https://github.com/LightForm-group/formable) | Formability analyses in Python | [Released](https://github.com/LightForm-group/matflow-formable) | [![PyPI version](https://img.shields.io/pypi/v/matflow-formable)](https://pypi.org/project/matflow-formable) |
| [DefDAP](https://github.com/MechMicroMan/DefDAP) | A python library for correlating EBSD and HRDIC data. | [Released](https://github.com/LightForm-group/matflow-defdap) | [![PyPI version](https://img.shields.io/pypi/v/matflow-defdap)](https://pypi.org/project/matflow-defdap) |
| [Abaqus](https://www.3ds.com/products-services/simulia/products/abaqus/) | Finite element analysis | In-progress | [![PyPI version](https://img.shields.io/pypi/v/matflow-abaqus)](https://pypi.org/project/matflow-abaqus) |
| [Neper](http://www.neper.info) | Polycrystal generation and meshing | [Released/In-progress](https://github.com/LightForm-group/matflow-neper) | [![PyPI version](https://img.shields.io/pypi/v/matflow-neper)](https://pypi.org/project/matflow-neper) |


### Example inputs/outputs 
| Label                   | Attributes                                                   | Output from tasks                         | Input to tasks                                               |
| ----------------------- | ------------------------------------------------------------ | ----------------------------------------- | ------------------------------------------------------------ |
| ODF                     | crystal_symmetry<br />speciment_symmetry<br />euler_angles<br />euler_angle_labels<br />weights<br />orientation_coordinate_system | get_model_texture<br />estimate_ODF<br /> | sample_texture                                               |
| microstructure_seeds    | position<br />**orientations**<br />grid_size<br />phase_label | generate_microstructure_seeds             | generate_volume_element                                      |
| orientations            | euler_angles<br />euler_angle_labels<br />orientation_coordinate_system | sample_texture                            | generate_volume_element                                      |
| volume_element          | grid<br />size<br />origin<br />**orientations**<br />grain_orientation_idx<br />grain_phase_label_idx<br />phase_labels<br />voxel_grain_idx<br />voxel_homogenization_idx | generate_volume_element                   | visualise_volume_element<br />simulate_volume_element_loading |
| load_case               | total_time<br />num_increments<br />def_grad_aim<br />def_grad_rate<br />stress<br />rotation | generate_load_case                        | simulate_volume_element_loading                              |
| volume_element_response | ...                                                          | simulate_volume_element_loading           |                                                              |

## Specifying default run options

Default run options (i.e. options passed to the scheduler) can be specified in a few ways. Firstly, within the workflow file, `run_options` specified at the top-level will be used for any tasks that do not have a `run_options` specified. If a task *does* have a `run_options` key specified, the global `run_options` will not be used at all for that task.

Additionally, you can specify default run options in the MatFlow configuration file (`config.yml`, by default generated in `~/.matflow`) with the options `default_run_options` and `default_sticky_run_options`. The "sticky" defaults are merged with any run options specified in the workflow file (with workflow-specified options taking precedence), whereas the "non-sticky" defaults are only used if no run options are supplied for a task. If no run options are supplied for a task, then both the "sticky" and "non-sticky" defaults will be used (with the "non-sticky" defaults taking precedence over the "sticky" defaults). Similar keys exist for task preparation and processing run options: `default_preparation_run_options`, `default_sticky_preparation_run_options` and `default_processing_run_options`, `default_sticky_processing_run_options`.
