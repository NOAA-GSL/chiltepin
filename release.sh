#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

# Pre-release validation script for Chiltepin
# Validates package is ready for release via GitHub Actions CI
#
# Usage:
#   ./release.sh check           - Run all pre-release checks
#   ./release.sh clean           - Clean build artifacts
#
# Release process (after checks pass):
#   1. TestPyPI: GitHub Actions → Test Release to TestPyPI → Run workflow
#   2. Production: git tag v<VERSION> && git push origin v<VERSION>

set -e  # Exit on error

PACKAGE_NAME="chiltepin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_dependencies() {
    log_info "Checking dependencies..."
    
    local missing_deps=()
    
    if ! command -v python &> /dev/null; then
        missing_deps+=("python")
    fi

    if ! python -c "import pytest" 2>/dev/null; then
        missing_deps+=("pytest")
    fi

    if ! python -c "import build" 2>/dev/null; then
        missing_deps+=("build")
    fi

    if ! python -c "import twine" 2>/dev/null; then
        missing_deps+=("twine")
    fi

    if ! python -c "import sphinx" 2>/dev/null; then
        missing_deps+=("sphinx")
    fi

    # Check for toml parsing (tomllib is built-in for Python 3.11+, otherwise need toml)
    if ! python -c "import tomllib" 2>/dev/null && ! python -c "import toml" 2>/dev/null; then
        missing_deps+=("toml or tomllib")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Install with: pip install -e \".[test,docs,release]\""
        exit 1
    fi

    log_success "All dependencies available"
}

# Get version from pyproject.toml
get_version() {
    # Try tomllib (Python 3.11+)
    python <<'EOF' 2>/dev/null && return
try:
    import tomllib
    with open('pyproject.toml', 'rb') as f:
        print(tomllib.load(f)['project']['version'])
except Exception:
    exit(1)
EOF

    # Try toml package (Python 3.10)
    python <<'EOF' 2>/dev/null && return
try:
    import toml
    with open('pyproject.toml', 'r') as f:
        print(toml.load(f)['project']['version'])
except Exception:
    exit(1)
EOF

    # Fallback to sed (no pipe needed)
    local version
    version=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
    if [ -n "$version" ]; then
        echo "$version"
        return 0
    else
        return 1
    fi
}

# Clean build artifacts
clean_build() {
    log_info "Cleaning build artifacts..."
    rm -rf dist/ build/ .eggs/
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    log_success "Build artifacts cleaned"
}

# Run tests (informational only - requires proper config)
run_tests() {
    log_info "Running basic test discovery..."
    log_warning "Note: Full test suite requires proper configuration"
    log_info "Run tests manually with config before release:"
    echo "  pytest --config=path/to/config.yaml"

    # Just check that pytest can collect tests
    if python -m pytest tests/ --collect-only -q > /dev/null 2>&1; then
        log_success "Test collection successful"
        return 0
    else
        log_warning "Test collection failed - check test files for syntax errors"
        return 1
    fi
}

# Build documentation
build_docs() {
    log_info "Building documentation..."

    # Clean old docs
    rm -rf docs/_build

    # Build docs with sphinx (capture output while preserving exit code)
    LC_ALL=C python -m sphinx -b html docs docs/_build -q -W --keep-going 2>&1 | tee /tmp/sphinx-build.log
    local sphinx_exit_code=${PIPESTATUS[0]}

    if [ $sphinx_exit_code -eq 0 ]; then
        log_success "Documentation built successfully"
        return 0
    else
        log_error "Documentation build failed"
        log_info "See /tmp/sphinx-build.log for details"
        return 1
    fi
}

# Build package
build_package() {
    log_info "Building package..."
    python -m build --sdist --wheel
    log_success "Package built successfully"
}

# Check package with twine
check_package() {
    log_info "Checking package with twine..."
    python -m twine check dist/*
    log_success "Package validation passed"
}

# Show package contents
show_package_info() {
    local version=$(get_version)
    log_info "Package information:"
    echo "  Name: ${PACKAGE_NAME}"
    echo "  Version: ${version}"
    echo ""
    log_info "Distribution files:"
    ls -lh dist/
}

# Show release checklist and next steps
show_release_guide() {
    local version=$(get_version)

    echo ""
    log_success "═══════════════════════════════════════════════════"
    log_success "   Package v${version} is ready for release!"
    log_success "═══════════════════════════════════════════════════"
    echo ""

    log_info "Pre-release checklist:"
    echo "  [ ] CHANGELOG.md is updated with release notes"
    echo "  [ ] Version number is correct in pyproject.toml (${version})"
    echo "  [ ] All tests pass: pytest --config=path/to/config.yaml"
    echo "  [ ] Documentation builds without errors (✓)"
    echo "  [ ] Package metadata is valid (✓)"
    echo "  [ ] All changes are committed to git"
    echo ""

    log_info "Test release to TestPyPI:"
    echo "  1. Go to: https://github.com/NOAA-GSL/chiltepin/actions"
    echo "  2. Select: 'Test Release to TestPyPI' workflow"
    echo "  3. Click: 'Run workflow' button"
    echo "  4. Test install: pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ chiltepin==${version}"
    echo ""

    log_info "Production release to PyPI:"
    echo "  1. Create and push tag:"
    echo "     git tag -a v${version} -m \"Release v${version}\""
    echo "     git push origin v${version}"
    echo "  2. GitHub Actions will automatically:"
    echo "     - Run validation checks"
    echo "     - Build and publish to PyPI"
    echo "     - Create GitHub release with notes"
    echo "  3. Monitor: https://github.com/NOAA-GSL/chiltepin/actions"
    echo ""

    log_warning "Note: Ensure PyPI trusted publishing is configured"
    echo "  https://pypi.org/manage/account/publishing/"
    echo ""
}

# Main command dispatcher
main() {
    cd "$SCRIPT_DIR"
    
    local command="${1:-check}"
    
    case "$command" in
        check)
            log_info "Running pre-release validation..."
            echo ""

            check_dependencies

            run_tests || true  # Informational only - don't fail on error
            echo ""

            if ! build_docs; then
                log_error "Documentation must build before release"
                exit 1
            fi
            echo ""

            clean_build
            build_package
            check_package
            show_package_info
            show_release_guide
            ;;

        clean)
            clean_build
            ;;

        help|--help|-h|*)
            echo "Chiltepin Pre-Release Validation"
            echo ""
            echo "This script validates the package is ready for release."
            echo "Actual releases are handled by GitHub Actions workflows."
            echo ""
            echo "Usage: $0 <command>"
            echo ""
            echo "Commands:"
            echo "  check      - Run all pre-release checks (default)"
            echo "  clean      - Clean build artifacts"
            echo "  help       - Show this help message"
            echo ""
            echo "Release workflow:"
            echo "  1. Run: ./release.sh check"
            echo "  2. Test: Trigger 'Test Release to TestPyPI' workflow in GitHub Actions"
            echo "  3. Release: git tag v<VERSION> && git push origin v<VERSION>"
            echo ""
            ;;
    esac
}

main "$@"
