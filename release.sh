#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

# Release automation script for Chiltepin
# Usage:
#   ./release.sh check           - Validate package without uploading
#   ./release.sh test            - Upload to TestPyPI
#   ./release.sh release         - Upload to production PyPI
#   ./release.sh clean           - Clean build artifacts

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
    python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])" 2>/dev/null || \
    python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])" 2>/dev/null || \
    grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'
}

# Clean build artifacts
clean_build() {
    log_info "Cleaning build artifacts..."
    rm -rf dist/ build/ src/*.egg-info .eggs/
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    log_success "Build artifacts cleaned"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    if python -m pytest tests/ -q --tb=line; then
        log_success "All tests passed"
        return 0
    else
        log_error "Tests failed"
        return 1
    fi
}

# Build documentation
build_docs() {
    log_info "Building documentation..."

    # Clean old docs
    rm -rf docs/_build

    # Build docs with sphinx
    if LC_ALL=C python -m sphinx -b html docs docs/_build -q -W --keep-going 2>&1 | tee /tmp/sphinx-build.log; then
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
    echo ""
    log_info "Source distribution contents (first 20 files):"
    tar -tzf dist/${PACKAGE_NAME}-${version}.tar.gz | head -20
}

# Upload to TestPyPI
upload_test() {
    log_warning "Uploading to TestPyPI..."
    echo ""
    log_info "You will need TestPyPI credentials."
    log_info "Register at: https://test.pypi.org/account/register/"
    log_info "Get API token at: https://test.pypi.org/manage/account/token/"
    echo ""
    read -p "Continue with upload? (y/N): " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Upload cancelled"
        exit 0
    fi
    
    python -m twine upload --repository testpypi dist/*
    
    local version=$(get_version)
    log_success "Upload to TestPyPI complete!"
    echo ""
    log_info "Test installation with:"
    echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple ${PACKAGE_NAME}==${version}"
}

# Upload to production PyPI
upload_release() {
    local version=$(get_version)
    
    log_warning "═══════════════════════════════════════════════════"
    log_warning "   PRODUCTION RELEASE TO PYPI"
    log_warning "═══════════════════════════════════════════════════"
    echo ""
    log_warning "This will upload version ${version} to production PyPI!"
    log_warning "This action CANNOT be undone!"
    echo ""
    log_info "Checklist before proceeding:"
    echo "  [ ] CHANGELOG.md is updated"
    echo "  [ ] Version number is correct in pyproject.toml"
    echo "  [ ] All tests pass"
    echo "  [ ] Documentation builds without errors"
    echo "  [ ] Package tested on TestPyPI"
    echo "  [ ] Git tag created: git tag -a v${version} -m 'Release v${version}'"
    echo "  [ ] Changes committed and pushed"
    echo ""
    read -p "Have you completed all checklist items? (y/N): " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Release cancelled"
        exit 0
    fi
    
    echo ""
    log_warning "Last chance to abort!"
    read -p "Type the version number '${version}' to confirm: " -r
    if [[ "$REPLY" != "$version" ]]; then
        log_error "Version mismatch. Release cancelled."
        exit 1
    fi
    
    log_info "Uploading to PyPI..."
    python -m twine upload dist/*
    
    log_success "═══════════════════════════════════════════════════"
    log_success "   Release ${version} published to PyPI!"
    log_success "═══════════════════════════════════════════════════"
    echo ""
    log_info "Next steps:"
    echo "  1. Push git tag: git push origin v${version}"
    echo "  2. Create GitHub release: https://github.com/NOAA-GSL/chiltepin/releases/new"
    echo "  3. Verify installation: pip install ${PACKAGE_NAME}==${version}"
}

# Main command dispatcher
main() {
    cd "$SCRIPT_DIR"
    
    local command="${1:-help}"
    
    case "$command" in
        check)
            log_info "Running pre-release checks..."
            check_dependencies
            
            if ! run_tests; then
                log_error "Tests must pass before release"
                exit 1
            fi

            if ! build_docs; then
                log_error "Documentation must build before release"
                exit 1
            fi

            clean_build
            build_package
            check_package
            show_package_info

            log_success "All checks passed! Package is ready for release."
            log_info "Next steps:"
            echo "  - Test on TestPyPI: ./release.sh test"
            echo "  - Release to PyPI:  ./release.sh release"
            ;;

        test)
            log_info "Preparing TestPyPI upload..."
            check_dependencies

            if ! run_tests; then
                log_error "Tests must pass before TestPyPI upload"
                exit 1
            fi

            if ! build_docs; then
                log_error "Documentation must build before TestPyPI upload"
                exit 1
            fi

            clean_build
            build_package
            check_package
            show_package_info
            upload_test
            ;;

        release)
            log_info "Preparing production release..."
            check_dependencies

            if ! run_tests; then
                log_error "Tests must pass before release"
                exit 1
            fi

            if ! build_docs; then
                log_error "Documentation must build before release"
                exit 1
            fi

            clean_build
            build_package
            check_package
            show_package_info
            upload_release
            ;;

        clean)
            clean_build
            ;;

        help|--help|-h|*)
            echo "Chiltepin Release Automation"
            echo ""
            echo "Usage: $0 <command>"
            echo ""
            echo "Commands:"
            echo "  check      - Run all checks and build package (no upload)"
            echo "  test       - Upload to TestPyPI for testing"
            echo "  release    - Upload to production PyPI (use with caution!)"
            echo "  clean      - Clean build artifacts"
            echo "  help       - Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 check            # Validate everything locally"
            echo "  $0 test             # Deploy to TestPyPI"
            echo "  $0 release          # Deploy to production PyPI"
            ;;
    esac
}

main "$@"
