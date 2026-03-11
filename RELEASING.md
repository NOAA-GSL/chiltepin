# Release Process Guide

This guide documents the process for releasing new versions of Chiltepin to PyPI.

Releases are automated via **GitHub Actions workflows**. The `release.sh` script validates package readiness locally but does not trigger automated workflows.

## Prerequisites

### Local Development Tools

Install release tools for local validation:
```bash
pip install -e ".[test,docs,release]"
```

### One-Time CI Setup

Configure **Trusted Publishing (OIDC)** for secure, token-free authentication:

1. **PyPI Trusted Publishing:**
   - Go to https://pypi.org/manage/account/publishing/
   - Add GitHub publisher:
     - **Owner:** `NOAA-GSL`
     - **Repository:** `chiltepin`
     - **Workflow:** `release.yml`
     - **Environment:** `pypi`

2. **TestPyPI Trusted Publishing:**
   - Go to https://test.pypi.org/manage/account/publishing/
   - Add GitHub publisher:
     - **Owner:** `NOAA-GSL`
     - **Repository:** `chiltepin`
     - **Workflow:** `test-release.yml`
     - **Environment:** `testpypi`

3. **GitHub Environments:**
   - Go to repository Settings → Environments
   - Create `pypi` environment (optionally add protection rules requiring reviewers)
   - Create `testpypi` environment (no protection needed)

## Release Workflow

### 1. Prepare Release Changes

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.2.0"  # Increment version (see Versioning section)
   ```

2. **Update CHANGELOG.md:**
   - Add new version section with date
   - Document all changes since last release
   - Follow [Keep a Changelog](https://keepachangelog.com/) format

### 2. Validate Package Locally

**Before committing**, validate everything is ready:

```bash
./release.sh check
```

This validates:
- ✅ All dependencies are installed
- ✅ Test collection works (warns about needing config for full tests)
- ✅ Documentation builds without errors
- ✅ Package builds successfully
- ✅ Package metadata passes twine checks
- ✅ Displays release checklist

**Complete all checklist items:**
- Run full test suite with config: `pytest --config=path/to/config.yaml`
- Fix any issues found
- Review the version number is correct

### 3. Commit Release Changes

Once validation passes, commit the changes:

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "Bump version to 0.2.0"
git push origin main  # or your feature branch
```

### 4. Test on TestPyPI

Test the complete release process using GitHub Actions:

1. Go to https://github.com/NOAA-GSL/chiltepin/actions
2. Select **"Test Release to TestPyPI"** workflow
3. Click **"Run workflow"**
4. Select your branch (or `main`)
5. Click **"Run workflow"** button
6. Monitor workflow execution

**Important:** The workflow verifies that the test suite successfully ran on the commit. If tests haven't run (or failed), the workflow will be blocked. This ensures only tested code is published, even to TestPyPI.

Once successful, test installation:

```bash
# Create a test environment
python -m venv /tmp/test-chiltepin
source /tmp/test-chiltepin/bin/activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ chiltepin==0.2.0

# Test basic functionality
python -c "import chiltepin; print(chiltepin.__version__)"
python -c "import chiltepin.tasks; import chiltepin.configure"

# Deactivate and clean up
deactivate
rm -rf /tmp/test-chiltepin
```

### 5. Create and Push Git Tag

Once TestPyPI testing is successful, create and push the version tag:

```bash
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0
```

**This automatically triggers the production release workflow!**

### 6. Monitor Automated Release

The tag push triggers the **"Release to PyPI"** workflow automatically:

1. Go to https://github.com/NOAA-GSL/chiltepin/actions
2. Find the running "Release to PyPI" workflow
3. Monitor execution (typically 2-3 minutes)

The workflow will:
- ✅ Verify test suite passed on this commit
- ✅ Run linting checks
- ✅ Build documentation
- ✅ Build distribution packages
- ✅ Validate with twine
- ✅ Publish to PyPI (via OIDC)
- ✅ Create GitHub Release with auto-generated notes

**Important:** The workflow verifies that the test suite successfully ran on the tagged commit. If tests haven't run (or failed), the release will be blocked. This ensures only tested code is released.

### 7. Post-Release Verification

1. **Verify PyPI release:**
   - Check https://pypi.org/project/chiltepin/
   - Verify correct version appears

2. **Verify GitHub release:**
   - Check https://github.com/NOAA-GSL/chiltepin/releases
   - Review auto-generated release notes
   - Edit if needed to add highlights or breaking changes

3. **Test production installation:**
   ```bash
   pip install --upgrade chiltepin
   python -c "import chiltepin; print(chiltepin.__version__)"
   ```

4. **Announce:**
   - Update README badges if needed
   - Announce in relevant channels
   - Update project documentation if needed

## Release Script Commands

The `release.sh` script is a **validation tool** for local pre-release checks. Actual releases are handled by GitHub Actions.

### `./release.sh check` (default command)

Runs all validation checks locally:
- Validates dependencies are installed
- Runs test collection check (warns that full tests need config)
- Builds documentation (must pass)
- Builds package distributions
- Validates package with twine
- Shows pre-release checklist and next steps

**Use this** before triggering any release workflow to ensure everything is ready.

### `./release.sh clean`

Removes all build artifacts:
- `dist/` directory
- `build/` directory 
- `*.egg-info` directories (anywhere in project)
- `__pycache__` directories
- `*.pyc` files

**Use this** to clean up between builds or when troubleshooting build issues.

## Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version (1.0.0): Incompatible API changes
- **MINOR** version (0.1.0): Add functionality (backwards-compatible)
- **PATCH** version (0.0.1): Bug fixes (backwards-compatible)

## Troubleshooting

### Workflow fails with authentication error

**If using Trusted Publishing (OIDC):**
- Verify PyPI/TestPyPI publisher configuration matches:
  - Repository: `NOAA-GSL/chiltepin`
  - Workflow filename: `release.yml` or `test-release.yml`
  - Environment name: `pypi` or `testpypi`
- Check GitHub environment names exist in Settings → Environments
- Ensure workflow has `id-token: write` permission (already configured)
- Check organization settings don't block OIDC tokens

**Alternative: Use API tokens instead:**
1. Uncomment token-based authentication in workflow files
2. Create API tokens at PyPI/TestPyPI
3. Add to repository secrets:
   - `PYPI_API_TOKEN` for production
   - `TEST_PYPI_API_TOKEN` for testing

### "Package already exists" on PyPI

- You cannot replace or delete a version on PyPI once published
- Must increment version number and release again
- Each version can only be uploaded once (this is by design)

### Documentation build fails

- Run `./release.sh check` locally to see detailed errors
- Check manually: `cd docs && make clean html`
- Look for Sphinx warnings/errors in output
- Common issues: missing imports, broken links, invalid RST syntax

### Release workflow fails: "Test suite must pass before release"

Both release workflows (production and TestPyPI) validate that the test suite passed on the commit before proceeding:

**To resolve:**
1. Ensure tests ran on the commit you're releasing:
   - Check: https://github.com/NOAA-GSL/chiltepin/actions
   - Look for "TestSuite" workflow run on your commit
2. If tests haven't run, trigger them:
   ```bash
   gh workflow run test-suite.yaml --ref main
   # Or for your branch:
   gh workflow run test-suite.yaml --ref your-branch-name
   ```
3. Wait for tests to complete successfully
4. Try the release again:
   - For TestPyPI: Re-run the workflow manually
   - For production: Delete and recreate the tag:
     ```bash
     git tag -d v0.2.0          # Delete local tag
     git push origin :v0.2.0     # Delete remote tag
     git tag -a v0.2.0 -m "Release version 0.2.0"
     git push origin v0.2.0
     ```

**Why this check exists:** Ensures only tested code is published. Since the test suite requires a container environment with config files, it's not re-run in release workflows but validated to have passed.

### Tests fail in workflow

The workflows run only basic linting and test collection checks since full tests require configuration files:
- Linting failures indicate code style issues: run `ruff check src/ tests/` locally
- Test collection failures indicate syntax errors in test files
- **Always run full test suite locally** before releasing: `pytest --config=path/to/config.yaml`

### Package validation fails

- Verify `pyproject.toml` metadata is correct (especially `license`, `description`, etc.)
- Run `twine check dist/*` locally after `./release.sh check`
- Check for missing required fields or incorrect formats
- Review build output for warnings

### Workflow doesn't trigger

- Verify tag format matches pattern: `v*.*.*` (e.g., `v0.1.0`)
- Check workflows are enabled: Settings → Actions → General
- Look for workflow runs: Actions tab (may show as skipped if pattern doesn't match)

## Security

### Trusted Publishing (OIDC)

**This is the recommended and default authentication method.**

Benefits:
- ✅ No long-lived secrets/tokens stored
- ✅ Tokens are ephemeral (valid only for single workflow run)
- ✅ Cannot be stolen from GitHub secrets
- ✅ Recommended by PyPI as best practice
- ✅ Reduced token management overhead

### Alternative: API Tokens

If OIDC is unavailable or organization policy requires tokens:

1. **Never commit tokens or `.pypirc`** to git
2. **API tokens are secrets** - treat them like passwords
3. Store tokens only in GitHub repository secrets
4. Scope tokens to specific projects when possible
5. Rotate tokens periodically (every 6-12 months)
6. Revoke tokens immediately if compromised

### Workflow Security

- Workflows use `contents: write` permission (for creating releases)
- Workflows use `id-token: write` permission (for OIDC)
- Production releases can optionally require approval via environment protection rules
- Consider requiring reviews for the `pypi` environment in sensitive projects

## GitHub Actions Workflows

Release workflows are located in `.github/workflows/`:

### `release.yml` - Production Release
- **Trigger:** Automatically on version tag push (`v*.*.*`)
- **Purpose:** Publish to production PyPI and create GitHub release
- **Environment:** `pypi`
- **Test validation:** Enforces that test suite passed on the tagged commit (blocking)

### `test-release.yml` - Test Release  
- **Trigger:** Manual via workflow_dispatch
- **Purpose:** Publish to TestPyPI for testing
- **Environment:** `testpypi`
- **Test validation:** Enforces that test suite passed on the commit (blocking)

Both workflows:
- Verify test suite passed on commit (required, blocking)
- Run linting checks (ruff)
- Build documentation (must pass)
- Build distributions (wheel + sdist)
- Validate with twine
- Publish using OIDC (trusted publishing)

## Manual Release (Emergency Fallback)

If GitHub Actions are unavailable, you can release manually:

```bash
# Validate package
./release.sh check

# Build distributions
python -m build

# Upload to TestPyPI (requires API token in ~/.pypirc)
twine upload --repository testpypi dist/*

# Upload to PyPI (requires API token in ~/.pypirc)
twine upload dist/*

# Create GitHub release manually via web UI
```

**Note:** This requires API tokens configured in `~/.pypirc` since OIDC only works with GitHub Actions.
