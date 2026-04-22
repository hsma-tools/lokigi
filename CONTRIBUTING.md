# Contributing

We welcome contributions to `lokigi`. You can either:

* [Create a GitHub issue](https://github.com/hsma-tools/lokigi/issues).
* [Fork the repository](https://github.com/hsma-tools/lokigi/fork) and create a pull request.

This document contains guidance for working on this repository. Please be respectful and considerate - see the`CODE_OF_CONDUCT.md`.

<br>


## Useful tips
- any new plotting methods added to site_solutions.py should be placed in lokigi.mixins.site_solution_plots.py
    - first check whether your new method needs a brand new mixin class, or whether it fits logically into one of the existing mixins
        - if setting an entirely new mixin class up, make sure to then update the setup of SiteSolutionSet to import and use your new mixin class
    - you will then need to update _quarto.yml to pick up your new plotting method in the SiteSolutionSet section of the documentation

- any new solvers added to site.py should be placed in lokigi.mixins.site_solvers.py as a new class.
    - make sure to then update the setup of SiteProblem to import and use your new mixin class
    - you will then need to update _quarto.yml to pick up your new solver in the dev section of the documentation
    - hook up your solver to SiteProblem._solve_pmedian_pcenter_mclp_problem

## Updating the list of contributors

Any contributors to the repository should be recognised via `all-contributors`. If your name or contributions are missing from the README, or if you contributed in ways not captured by the current role emojis, then please feel free to update these. There are two ways to do this:

### 1. Via GitHub issues

This is the simplest option. Just create an issue like this example:

```
@all-contributors please add @githubuser for ...
```

Then list appropriate contribution types from [allcontributors.org/docs/en/emoji-key](https://allcontributors.org/docs/en/emoji-key) (e.g., code, review, doc, content, bug, ideas, infra).

### 2. Via the command line

Alternatively, you can update it from the command line. This may be preferable, as the bot will send emails to anyone tagged, and requires making pull requests into main (which may trigger various GitHub actions).

You'll need to install the [All-Contributors CLI tool](https://allcontributors.org/cli/installation/):

```
npm i -D all-contributors-cli
```

You can then run the following and select/enter relevant information when prompted:

```
npx all-contributors
```

If you want to remove specific contributions or people, edit the `.all-contributorsrc` file then run the following to regenerate the table in `README.md`. (Don't edit `README.md`, as it is just generated based on `.all-contributorsrc`).

```
npx all-contributors generate
```

<br>

## Development environment

### Python

A development environment is provided in `dev_environment/`. You can choose between:

* A conda environment (`environment.yml`).
* A virtualenv (`requirements.txt`).

You will also want to install the local lokigi package by running `pip install -e .`.

The conda environment will also install a suitable version of Python - if using virtualenv, you will need to configure this yourself.

This environment differs from `lokigi`'s dependencies (`pyproject.toml`), as it contains the packages needed to e.g., generate documentation, run tests, lint code, and build the package.

If you make changes to the development environment, please ensure you change it in all locations:

* [ ] `dev_environment/environment.yml`
* [ ] `dev_environment/requirements.txt`

## Documentation

The lokigi documentation is created using quarto and `quartodoc`. You can generate it locally by running:

```
quartodoc build
quarto render lokigi_docs
```

It is rendered via GitHub actions and hosted on GitHub pages. The action creates a Docker image hosted on GitHub Container Registry. This makes it more efficient, as it doesn't need to rebuild the environment when no changes have been made to the packages installed.

To test rendering the quarto site in the docker container locally...

Build image:

```
sudo docker build -t lokigi .
```

Render quarto project inside container:

```
docker run --rm lokigi quarto render
```

<br>

## Linting

We use Black to auto-format the lokigi package, setting the maximum line length to 79 to comply with PEP 8 - simply run:

```
black lokigi --line-length=79
```

We also run other linters to manually check and edit package style:

```
# Checks PEP8-style, basic errors and code complexity
flake8 lokigi

# Run flake8 on .ipynb files
nbqa flake8 examples

# Run flake8 on .qmd files
lintquarto -l flake8 -p lokigi_docs
```
