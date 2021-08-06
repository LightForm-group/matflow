# Change Log

## [0.2.22] - 2021.xx.xx

### Fixed

- Fix error message if an input mapper function has an unknown argument.

## [0.2.21] - 2021.06.06

### Added

- Allow passing a subset of the task input parameters to the output mapper function. Resolve [#102](https://github.com/LightForm-group/matflow/issues/102).
- Allow passing all iterations of an input parameter to a function mapper. Resolve [#104](https://github.com/LightForm-group/matflow/issues/104).
- Allow running an on-demand archive to an existing/completed workflow: `matflow archive path/to/workflow/directory ARCHIVE_NAME`. Resolve [#68](https://github.com/LightForm-group/matflow/issues/68).
- Allow specifying `default_metadata` in the `config.yml` file. Keys are merged with `metadata` specified in the workflow spec file. Resolve [#98](https://github.com/LightForm-group/matflow/issues/98).

### Fixed

- Save element resource usage (e.g. run time). Fix [#97](https://github.com/LightForm-group/matflow/issues/97).
- Fix bug when determining the "producing task" in an iteration pathway. Fix [#105](https://github.com/LightForm-group/matflow/issues/105).
- Fix bug when a file input parameter is specified with a `$HOME` tilde: `~/path/to/file`.

## [0.2.20] - 2021.05.12

### Added

- Add `Task.cleanup` attribute that can be used to optionally specify a list of glob patterns, representing file names to remove at the end of `Workflow.process_task_element`. Useful for removing very large simulation outputs that are not required after MatFlow has extracted the requested data.
- Add methods to `Element` object: `get_file_lines` and `print_file_lines`, which take a file name and a slice of lines to get or print.

### Changed

- Change working directory to element directory for invoking input/output/function mapper functions. This is required in some cases where a tool or script does not accept a file path as an argument.
- Allow specifying the `task_idx` directly when importing parameters. This overrides any specified `context`.

### Fixed

- Catch `ImportError` and `SyntaxError` when trying to load extensions.
- Import from the highest task index when importing a parameter that has been through a parameter-modifying task - fix [#103](https://github.com/LightForm-group/matflow/issues/103). The can be overrode by specifying a `task_idx` directly.

## [0.2.19] - 2021.04.12 (April 2021 - Fix 1)

### Fixed

- Fix type problem when input schema keys are specified "inline" in the task schema (e.g. as `CRC_file_path[file=True,save=False]`), in which the keys remain as type `str`, when they should be `bool`.
- Fix problem when an imported parameter is used in a task that is iterated.

## [0.2.18] - 2021.04.10 (April 2021)

### Fixed

- Fix misleading error message when a task parameter specified as a file path does not actually exist as a file.
- Fix bug where if all possible dependency pathways are circularly dependent, this is not caught by MatFlow. Fix [#88](https://github.com/LightForm-group/matflow/issues/88).
- Fix issue with accessing parameter data with dot-notation via their "safe names". Fix [#87](https://github.com/LightForm-group/matflow/issues/87).

### Added

- Add new parameter key `ignore_dependency_from`, which is a list of task names. This allows us to exclude tasks when considering the dependencies of this parameter. Fix [#89](https://github.com/LightForm-group/matflow/issues/89).
- Allow embedding file-path inputs (inputs that are text files) into the HDF5 file. Fix [#86](https://github.com/LightForm-group/matflow/issues/86).
- Add `Task.unique_name` property which adds on the non-trivial `Task.context` to `Task.name`.
- Tasks can be accessed from the task list via dot-notation. Fix [#90](https://github.com/LightForm-group/matflow/issues/90).
- Add `Task.elements_idx` property to retrieve to correct `elements_idx` dict for that task.
- Add new exception type: `ParameterImportError`.
- Add ability to import parameters from existing workflows. Fix [#30](https://github.com/LightForm-group/matflow/issues/30)

### Changed

- Non-trivial task contexts are now part of the task directory name to help distinguish task directories where multiple contexts are used. Fix [#50](https://github.com/LightForm-group/matflow/issues/50).
- Add `context` argument to `Workflow.get_input_tasks` and `Workflow.get_output_tasks`.

## [0.2.17] - 2021.02.15

### Fixed

- Fix issue [#82](https://github.com/LightForm-group/matflow/issues/82) where the default group is not defined in the `Workflow.element_idx` for tasks where no local inputs are defined.

### Added

- Add support for flexible positioning of parameter-modifying tasks ([#81](https://github.com/LightForm-group/matflow/issues/81))

## [0.2.16] - 2021.02.05

### Fixed

- Bump hpcflow to v0.1.13 to fix #80 and then to v0.1.14 to fix a database locking issue and a bug with choosing the correct working directories.

## [0.2.15] - 2021.01.18

### Changed

- Change an Exception to a warning in `Workflow.get_element_data` to allow manually deleting element data without corrupting.

## [0.2.14] - 2021.01.17

### Added

- Add method `Task.get_elements_from_iteration(iteration_idx)`.

## [0.2.13] - 2020.12.17

### Fixed 

- Fix bug when populating `Workflow.elements_idx` for more than two iterations.

## [0.2.12] - 2020.12.16

### Added

- Add `Workflow.figures` attribute for storing associated figure definitions.
- Add `Workflow.metadata` attribute for storing arbitrary metadata (will later be used for Zenodo archiving).
- Add various `Workflow` static methods to help with retrieving information in the viewer without loading the whole workflow via `hickle`.
- Add `get_task_schemas` to API to load the available task schemas without generating a workflow.
- Add `refresh` bool parameter to `Config.set_config`, to force a reload of the configuration.
- Support inputs as dependencies as well as outputs.
- Support "parameter modifying" tasks (a task which outputs a parameter that is also an input to that task).
- Add `iterate_run_options` to Workflow.
- Add new methods for finding dependent and dependency tasks/parameters, upstream/downstream parameter values associated with a given element.
- Add input option: `include_all_iterations`. If True, inputs from all iterations are passed to input map functions.

### Fixed

- Only save input/output map files if they exist!
- Fix bug in propagating groups correctly
- Various code formatting issues
- Fix failure to raise on invalid schemas.
- Fix bug when the same file is to be saved from multiple output maps.

### Changed
- Redo task sorting algorithm such that minimal ordering changes are made.
- Set `stats` bool to False by default.
- Bump hpcflow version to v0.1.12.

## [0.2.11] - 2020.09.29

### Fixed

- Resolve `~` in task schema and software file paths specified in the configuration file.

## [0.2.10] - 2020.09.29

### Fixed 

- Fix if a function mapper function does not return anything.

## [0.2.9] - 2020.09.17

### Added

- Add scripting module for generating Python source scripts.
- Default run options can be specified in the MatFlow configuration file for task, preparation and processing jobs using both "sticky" and "non-sticky" keys: `default_run_options`, `default_sticky_run_options`, `default_preparation_run_options`, `default_sticky_preparation_run_options`, `default_processing_run_options` and `default_sticky_processing_run_options`. The "sticky" defaults are always applied (but workflow-specified run options take precedence), whereas the "non-sticky" defaults are only applied if a task has no workflow-specified run options.

## [0.2.8] - 2020.09.01

### Changed
- Add `version_info` to `Software.__repr__` method
- Validate source maps after missing schema check

### Fixed 
- Remove vestigial and buggy line in `construction.get_element_idx` which would lead to enormous memory usage for large sequences.

## [0.2.7] - 2020.08.18

### Added
- Default values can be specified for output map options within the schema
- Default values can be specified for task input parameters within the schema
- Depending on the inputs defined, different commands can be run, via "command pathway" definitions in the schema implementations.

### Changed

- Uses `hickle` version 4.
- Group structure in workflow HDF5 file has changed (backwards-incompatible); element data is more conveniently organised for inspecting the HDF5 file manually.

### Fixed

- Fix problem when a task input key includes slashes.

## [0.2.6] - 2020.07.08

### Added

- Add alternate scratch feature to allow a given task to be executed within a separate temporary directory.

### Fixed

- Fix bug if specifying `merge_priority` on the default group.

### Changed

- Bump hpcflow to v0.1.10

## [0.2.5] - 2020.06.27

### Fixed

- Fix copying of profile file to the workflow directory when the profile file path is not in the current working directory.

## [0.2.4] - 2020.06.26

### Changed

- Fix dependency `hickle` version for now, until we can assess requirements for jumping to version 4.

## [0.2.3] - 2020.06.26

### Changed

- Files generated by input maps are only saved into the workflow file if explicitly requested with `save: true`.

### Fixed

- Fix bug in `SourcesPreparation.get_formatted_commands` that appears if there are no commands.

## [0.2.2] - 2020.06.09

### Changed

- Improved Dropbox authorization flow. 
- Bump hpcflow to v0.1.9

## [0.2.1] - 2020.06.09

### Fixed

- Fix bug in reading `default_preparation_run_options` and `default_processing_run_options` dicts from the config file.

## [0.2.0] - 2020.06.09

### Added

- Add a `Workflow.history` attribute that tracks when the workflow was modified. It also stores pertinent software versions.
- Add a CLI command `matflow validate` that runs through the task schema and extension validation.
- Add a CLI command `matflow kill`, which kills all executing and pending tasks.
- Added configuration option `prepare_process_scheduler_options` to specify scheduler options for the prepare and process tasks.
- matflow profile is stored as a `dict` in addition to a string representation of the profile file (both in the `Workflow.profile` attribute).

### Changed

- Module and function `jsonable.py` and `to_jsonable` renamed to `hicklable.py` and `to_hicklable`.
- Workflow and Task attributes in the workflow HDF5 file are now represented without leading underscores.
- Tasks with only a single element use the task directory directly instead of using an element sub-directory.
- Loading extensions and configuration files has been moved from the root `__init__` to separate modules.
- `make_workflow`, `submit_workflow`, `load_workflow`, `append_schema_source`, `prepend_schema_source` and `validate` can now be imported from the root level: `from matflow import make_workflow` etc.
- There are no longer unsightly global variables for `TASK_INPUT_MAP` etc. This functionality has been subsumed into the global `Config` class. This is tidier and provides a better place for some validation.
- Software key `sources` has been replaced by `environment`.
- hpcflow configuration directory is generated within the matflow configuration directory.
- Jobscript names refer to the task to which they prepare/execute/process
- hpcflow profile is passed as a `dict` to hpcflow. For information, the hpcflow profile is still dumped to a file.

## [0.1.3] - 2020.05.27

- New release for Zenodo archive.

## [0.1.2] - 2020.05.12

- Latest dev branch merged...

## [0.1.1] - 2020.05.07

### Fixed

- Added missing dependency.

## [0.1.0] - 2020.05.07

Initial release.
