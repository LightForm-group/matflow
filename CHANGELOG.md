# Change Log

## [0.2.0] - 2020.xx.xx

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
