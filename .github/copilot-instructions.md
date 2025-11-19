# GitHub Copilot Instructions for openms-python

## Repository Overview

`openms-python` is a Pythonic wrapper around pyOpenMS for mass spectrometry data analysis. The goal is to provide an intuitive, Python-friendly interface that makes working with mass spectrometry data feel natural for Python developers and data scientists.

**Key Principle**: Make pyOpenMS more Pythonic by wrapping verbose C++ bindings with intuitive Python APIs.

## Code Style and Conventions

### Python Style
- Follow PEP 8 conventions
- Use Black formatter with 100 character line length (configured in `pyproject.toml`)
- Target Python 3.8+ compatibility
- Use type hints for better IDE support and code clarity
- Prefer clear, descriptive names over abbreviations

### Wrapper Design Patterns

1. **Properties over getters/setters**: Use `@property` decorators instead of verbose get/set methods
   ```python
   # Good
   spec.retention_time
   # Avoid
   spec.getRT()
   ```

2. **Pythonic iteration**: Support Python's iteration protocols (`__iter__`, `__len__`, `__getitem__`)
   ```python
   for spec in experiment.ms1_spectra():
       print(spec.retention_time)
   ```

3. **Method chaining**: Return `self` from mutation methods to enable fluent interfaces
   ```python
   exp.filter_by_ms_level(1).filter_by_rt(100, 500)
   ```

4. **DataFrame integration**: Provide `to_dataframe()` and `from_dataframe()` methods for pandas interoperability

5. **Context managers**: Support `with` statements for file I/O operations

6. **Mapping interface for metadata**: Classes wrapping `MetaInfoInterface` should support dict-like access
   ```python
   feature["label"] = "sample_a"
   ```

### Class Naming Convention
- Wrapper classes use the `Py_` prefix (e.g., `Py_MSExperiment`, `Py_FeatureMap`)
- This distinguishes them from pyOpenMS classes while maintaining recognizability

### File Organization
- Core wrapper classes: `py_*.py` files (e.g., `py_msexperiment.py`, `py_featuremap.py`)
- I/O utilities: `io.py` and `_io_utils.py`
- Helper utilities: `_meta_mapping.py` for metadata handling
- Workflow helpers: `workflows.py` for high-level pipelines
- Example data: `examples/` directory contains sample files like `small.mzML`

## Testing Requirements

### Test Structure
- All tests in `tests/` directory
- Test files follow `test_*.py` naming convention
- Use pytest as the testing framework
- Aim for good coverage of wrapper functionality

### Running Tests
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest -v

# Run with coverage
pytest -v --cov=openms_python --cov-report=term-missing
```

### Test Patterns
- Test basic wrapper functionality (properties, methods)
- Test DataFrame conversions (to/from)
- Test file I/O (load/store operations)
- Test iteration and filtering
- Test method chaining
- Use `conftest.py` for shared fixtures

## Development Setup

### Installation
```bash
git clone https://github.com/openms/openms-python.git
cd openms-python
pip install -e ".[dev]"
```

### Dependencies
- **Core**: pyopenms (>=3.0.0), pandas (>=1.3.0), numpy (>=1.20.0)
- **Dev**: pytest, pytest-cov, black, flake8, mypy

### Code Formatting
```bash
# Format code with Black
black openms_python tests

# Check style with flake8
flake8 openms_python tests
```

## Key Architecture Patterns

### 1. Wrapper Pattern
Most classes wrap a corresponding pyOpenMS class and delegate to it while providing Pythonic interfaces:
```python
class Py_MSExperiment:
    def __init__(self, exp=None):
        self._exp = exp if exp is not None else oms.MSExperiment()
    
    @property
    def retention_time(self):
        return self._exp.getRT()
```

### 2. Factory Methods
Use class methods for alternative constructors:
```python
@classmethod
def from_file(cls, filepath):
    # Load from file and return new instance
    
@classmethod
def from_dataframe(cls, df):
    # Create from pandas DataFrame
```

### 3. Smart Filtering
Provide multiple ways to filter data:
- Method-based: `filter_by_rt(min_rt, max_rt)`
- Property-based: `rt_filter[min:max]`
- Iterator-based: `ms1_spectra()`, `ms2_spectra()`

### 4. Metadata Handling
Classes that wrap `MetaInfoInterface` should implement mapping protocol:
- `__getitem__`, `__setitem__`, `__delitem__`
- `__contains__`, `__iter__`, `__len__`
- `get()`, `pop()`, `update()` methods

## Common Tasks

### Adding a New Wrapper Class
1. Create a new `py_<classname>.py` file
2. Wrap the corresponding pyOpenMS class
3. Add Pythonic properties for common getters/setters
4. Implement `__len__`, `__iter__`, `__getitem__` if applicable
5. Add `to_dataframe()` and `from_dataframe()` if appropriate
6. Add `load()` and `store()` methods for file I/O
7. Write comprehensive tests in `tests/test_py_<classname>.py`
8. Update `__init__.py` to export the new class
9. Add examples to README.md

### Adding Helper Functions
- High-level workflow functions go in `workflows.py`
- I/O utilities go in `io.py` or `_io_utils.py`
- Metadata utilities go in `_meta_mapping.py`

### Documentation
- Add docstrings to all public classes and methods
- Include usage examples in docstrings
- Update README.md with new features
- Keep API reference section in README current

## Special Considerations

### Memory Management
- Be mindful of memory when working with large datasets
- Provide streaming alternatives for large files (see `stream_mzml`)
- Consider using generators for iteration over large collections

### pyOpenMS Compatibility
- The package depends on pyOpenMS >= 3.0.0
- When wrapping pyOpenMS classes, preserve all functionality
- Add convenience methods but don't remove or break existing capabilities

### Error Handling
- Provide clear, helpful error messages
- Validate inputs before passing to pyOpenMS
- Handle common edge cases (empty containers, missing files, etc.)

### Performance
- Wrapper overhead should be minimal
- Avoid unnecessary data copies
- Use NumPy arrays for peak data when possible
- Consider performance implications of DataFrame conversions

## Examples and Documentation

The README.md contains extensive examples. When adding new features:
1. Add code examples showing the improvement over pyOpenMS
2. Use "Before (pyOpenMS)" vs "After (openms-python)" format
3. Include practical use cases
4. Show integration with pandas/numpy when relevant

## CI/CD

The repository uses GitHub Actions for continuous integration:
- Workflow: `.github/workflows/integration-tests.yml`
- Runs on: Python 3.10 (configurable via matrix)
- Tests run automatically on push to main and on pull requests

## Contributing Guidelines

When contributing:
1. Make minimal, focused changes
2. Maintain backward compatibility unless explicitly breaking
3. Add tests for new functionality
4. Format code with Black
5. Ensure all tests pass
6. Update documentation as needed

## Questions or Issues?

- Check existing documentation in README.md
- Review existing wrapper implementations for patterns
- Look at test files for usage examples
- Open a discussion on GitHub for design questions
