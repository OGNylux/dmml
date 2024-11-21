# dstools
dstools is a collection of methods and datasets for teaching Data Science ad the St. Pölten University of Applied Sciences.

## Getting started

### New environment
Use conda to create a new environment using the required dependencies. We suggest using [Miniforge](https://github.com/conda-forge/miniforge) as your conda installer of choice.

Please use following command to create a new _dstools_ environment:

```shell
(base)$ conda env create -f environment.yml
(base)$ conda activate dstools
```

### Existing environment
You can also install all required dependencies in an existing environment:

```shell
   (base)$ conda activate foo-env
(foo-env)$ conda env update --file environment.yml
```
