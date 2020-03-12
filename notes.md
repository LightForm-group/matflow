Task inputs are "consumed" in one of two places:

1. In an input map, which translates inputs into files, that are in turn used by the commands
2. As command-line arguments for the commands themselves



## Ideas for later

<u>Online workflow viewer</u>

- Takes a `workflow.hdf5` file (e.g. from a public GitHub repo) and loads it with a simple web interface
- Inputs/outputs can be explored and data plotted.
- How would this be different from a local user interface (using a web framework)? Maybe very similar and useful (wouldn't need to install anything to explorer workflow -- a bit like Binder).