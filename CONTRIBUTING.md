# Contributing to Sign Language Detection

First off, thank you for considering contributing to Sign Language Detection! It's people like you that make it such a great tool.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone git@github.com:your-username/sign-language-detection.git
   cd sign-language-detection
   ```

3. Set up your development environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   pre-commit install
   ```

4. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Process

1. Write your code following our coding standards
2. Add or update tests as needed
3. Update documentation
4. Run the test suite
5. Submit a pull request

### Coding Standards

We use several tools to maintain code quality:

- **Black** for code formatting:
  ```bash
  black .
  ```

- **isort** for import sorting:
  ```bash
  isort .
  ```

- **flake8** for style guide enforcement:
  ```bash
  flake8 src tests
  ```

- **mypy** for type checking:
  ```bash
  mypy src tests
  ```

### Running Tests

Before submitting a pull request, make sure all tests pass:

```bash
pytest tests/
pytest --cov=src  # For coverage report
```

## Pull Request Process

1. Update the README.md with details of changes to the interface, if applicable
2. Update the documentation with any new features or changes
3. Ensure all tests pass and coverage remains high
4. The PR will be merged once you have the sign-off of at least one maintainer

### Pull Request Guidelines

- Use a descriptive title
- Follow the pull request template
- Link any related issues
- Describe your changes in detail
- List any dependencies that are required
- Include screenshots for UI changes
- Add notes on testing, if applicable

## Documentation

- Document all functions, classes, and modules using Google-style docstrings
- Update the README.md if you change functionality
- Comment complex algorithms or non-obvious solutions

### Example Docstring

```python
def function_name(param1: type1, param2: type2) -> return_type:
    """Short description.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ErrorType: When and why this error occurs
    """
    pass
```

## Project Structure

Please maintain the following project structure:

```
src/
├── data/           # Data loading and preprocessing
├── models/         # Model implementations
├── training/       # Training utilities
├── visualization/  # Visualization tools
└── utils/         # Helper functions

tests/             # Test files
docs/              # Documentation
notebooks/         # Jupyter notebooks
```

## Issue Reporting

### Bug Reports

When filing a bug report, please include:

1. Step-by-step description to reproduce the problem
2. Expected vs actual behavior
3. Python version and OS information
4. Complete error message and stack trace
5. Code sample or test case demonstrating the issue

### Feature Requests

For feature requests:

1. Describe the feature in detail
2. Explain why this feature would be useful
3. Provide examples of how it would be used
4. Discuss potential implementation approaches

## Release Process

1. Update version number in:
   - pyproject.toml
   - setup.py
   - docs/conf.py

2. Update CHANGELOG.md following Keep a Changelog format

3. Create a new release on GitHub with:
   - Version number as tag
   - Release notes
   - Binary attachments if applicable

## Questions?

Feel free to:

- Open an issue with questions
- Join our discussions
- Contact the maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
