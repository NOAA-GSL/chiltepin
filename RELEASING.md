# Release Process Guide

This guide documents the process for releasing new versions of Chiltepin to PyPI.

## Prerequisites

1. **Install release tools:**
   ```bash
   pip install -e ".[test,docs,release]"
   ```

2. **TestPyPI Account:**
   - Register at https://test.pypi.org/account/register/
   - Create API token at https://test.pypi.org/manage/account/token/
   - Save token in `~/.pypirc`:
     ```ini
     [testpypi]
     username = __token__
     password = pypi-YOUR-TEST-TOKEN-HERE
     ```

3. **PyPI Account:**
   - Register at https://pypi.org/account/register/
   - Create API token at https://pypi.org/manage/account/token/
   - Save token in `~/.pypirc`:
     ```ini
     [pypi]
     username = __token__
     password = pypi-YOUR-PRODUCTION-TOKEN-HERE
     ```

## Release Workflow

### 1. Prepare for Release

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.2.0"  # Increment version
   ```

2. **Update CHANGELOG.md:**
   - Add new version section
   - Document all changes
   - Follow [Keep a Changelog](https://keepachangelog.com/) format

3. **Commit changes:**
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "Bump version to 0.2.0"
   ```

### 2. Test Locally

Run the check command to validate everything:

```bash
./release.sh check
```

This will:
- ✅ Run all tests
- ✅ Build documentation
- ✅ Clean old builds
- ✅ Build new distributions
- ✅ Validate package with twine
- ✅ Show package contents

### 3. Test on TestPyPI

Upload to TestPyPI to verify the release process:

```bash
./release.sh test
```

Then test installation:

```bash
# Create a test environment
python -m venv /tmp/test-chiltepin
source /tmp/test-chiltepin/bin/activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple chiltepin==0.2.0

# Test basic functionality
python -c "import chiltepin; print(chiltepin.__version__)"

# Deactivate and clean up
deactivate
rm -rf /tmp/test-chiltepin
```

### 4. Create Git Tag

Once testing is successful, create and push a git tag:

```bash
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0
git push origin main
```

### 5. Release to Production PyPI

**⚠️ WARNING: This cannot be undone!**

```bash
./release.sh release
```

The script will:
- Run all tests (must pass)
- Show a checklist
- Require confirmation
- Upload to PyPI

### 6. Post-Release

1. **Create GitHub Release:**
   - Go to https://github.com/NOAA-GSL/chiltepin/releases/new
   - Select the tag (v0.2.0)
   - Copy changelog entry as release notes
   - Publish release

2. **Verify Installation:**
   ```bash
   pip install --upgrade chiltepin
   python -c "import chiltepin; print(chiltepin.__version__)"
   ```

3. **Announce:**
   - Update README badges if needed
   - Announce in relevant channels

## Release Script Commands

### `./release.sh check`
Runs all validation checks without uploading:
- Validates dependencies
- Runs test suite
- Builds documentation
- Builds package
- Checks with twine
- Shows package info

**Use this** during development to ensure package is ready.

### `./release.sh test`
Uploads to TestPyPI:
- Runs tests (must pass)
- Builds documentation (must pass)
- Builds package
- Uploads to test.pypi.org

**Use this** to test the package installation process.

### `./release.sh release`
Uploads to production PyPI:
- Runs tests (must pass)
- Builds documentation (must pass)
- Shows safety checklist
- Requires version confirmation
- Uploads to pypi.org

**Use this** only after thorough testing!

### `./release.sh clean`
Removes all build artifacts:
- dist/ directory
- build/ directory
- .egg-info directories
- __pycache__ directories
- .pyc files

## Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version (1.0.0): Incompatible API changes
- **MINOR** version (0.1.0): Add functionality (backwards-compatible)
- **PATCH** version (0.0.1): Bug fixes (backwards-compatible)

## Troubleshooting

### "Package already exists" on PyPI
- You cannot replace or delete a release on PyPI
- Increment version and release again
- Each version can only be uploaded once

### Tests fail
- Fix issues before releasing
- Do not skip test failures for production releases
- `release.sh` blocks uploads to both TestPyPI and PyPI if there are test failures

### Twine authentication fails
- Check `~/.pypirc` configuration
- Verify API token is correct
- Token must start with `pypi-`

### Import issues after installation
- Check package structure with `tar -tzf dist/*.tar.gz`
- Verify all necessary files are included
- Check `pyproject.toml` configuration

## Security

- **Never commit `.pypirc`** to git
- **API tokens are secrets** - treat them like passwords
- Consider using trusted publishing with GitHub Actions
- Rotate tokens periodically

## Automation with GitHub Actions

For future consideration, we could automate releases with GitHub Actions:

```yaml
# .github/workflows/release.yml
name: Release to PyPI
on:
  push:
    tags:
      - 'v*'
jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install build~=1.4.0 twine~=6.2.0
      - run: python -m build
      - run: twine upload dist/*
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
```
