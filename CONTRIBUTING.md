# Contributing

The preferred way to contribute to AcurusTrack is to fork the main repository on GitHub, then submit a “pull request” (PR) - as done for scikit-learn contributions:

- Fork the project repository: click on the ‘Fork’ button near the top of the page. This creates a copy of the code under your account on the GitHub user account. 

- Clone your fork of the scitime repo from your GitHub account to your local disk:
```$ git clone git@github.com:YourLogin/acurustrack.git
$ cd AcurusTrack
# Install library in editable mode:
$ pip install --editable .
```


- Create a branch to hold your development changes:

```
$ git checkout -b my-feature
```


and start making changes. Always use a feature branch. It’s good practice to never work on the masterbranch!

- Develop the feature on your feature branch on your computer, using Git to do the version control. When you’re done editing, add changed files using git add and then git commit files:
```
$ git add modified_files
$ git commit
```

- to record your changes in Git, then push the changes to your GitHub account with:
```
$ git push -u origin my-feature
```


- Follow GitHub instructions to create a pull request from your fork. 

Some quick additional notes:

- We try to follow the PEP8 guidelines (using flake8, ignoring codes E501 and F401).



## Help

If you're having trouble using this project, please start by reading the [`README.md`](README.md)
and searching for solutions in the existing open and closed issues.

## Code of Conduct

Please be sure to read and understand our [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
We work hard to ensure that our projects are welcoming and inclusive to as many
people as possible.

## Reporting Issues

If you have a bug report, please provide as much information as possible so that
we can help you out:

- Version of the project you're using.
- Code (or even better whole projects) which reproduce the issue.
- Steps which reproduce the issue.
- Screenshots, GIFs or videos (if relevant).
- Stack traces for crashes.
- Any logs produced.

## Making Changes

1. Fork this repository to your own account
2. Make your changes and verify that tests pass
3. Commit your work and push to a new branch on your fork
4. Submit a pull request
5. Participate in the code review process by responding to feedback

Once there is agreement that the code is in good shape, one of the project's
maintainers will merge your contribution.

To increase the chances that your pull request will be accepted:

- Follow the coding style
- Write tests for your changes
- Write a good commit message

## License

By contributing to this project, you agree that your contributions will be
licensed under its GNU GPL-3.0-or-later (LICENSE)
