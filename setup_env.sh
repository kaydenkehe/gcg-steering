#!/usr/bin/env bash

# Simple environment setup for the gcg-steering project.
# - Installs system build deps needed for tokenizers (pkg-config, OpenSSL dev)
# - Creates a local virtualenv (.venv) with python3
# - Ensures pip is up to date
# - Installs Rust (via rustup) if not present (needed to build tokenizers if no wheel)
# - Installs Python dependencies from requirements.txt

set -euo pipefail

REQUIRED_PYTHON="3.10"

version_ge() {
    [ "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1" ]
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "python3 not found on PATH. Please install Python ${REQUIRED_PYTHON}+ and retry."
        return 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')

    if ! version_ge "$PYTHON_VERSION" "$REQUIRED_PYTHON"; then
        echo "This script expects Python >= ${REQUIRED_PYTHON}, but found ${PYTHON_VERSION}."
        echo "Please install a newer python3 and retry."
        return 1
    fi

    return 0
}

install_system_deps() {
    # Install build essentials and OpenSSL headers for tokenizers builds
    if command -v apt-get >/dev/null 2>&1; then
        echo "Installing system build dependencies (pkg-config, libssl-dev, build-essential, curl, rustc, cargo)..."
        apt-get update -y
        apt-get install -y pkg-config libssl-dev build-essential curl rustc cargo
    else
        echo "WARNING: apt-get not found; please install pkg-config, libssl-dev, build-essential, and curl manually."
    fi
}

setup_venv() {
    if [ ! -d ".venv" ]; then
        echo "Creating virtualenv in .venv ..."
        python3 -m venv .venv
    else
        echo "Using existing virtualenv in .venv ..."
    fi

    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo "Activated virtualenv: $(python -V)"
}

ensure_rust() {
    if command -v rustc &> /dev/null; then
        echo "Rust compiler already installed: $(rustc --version)"
        return 0
    fi
    echo "Rust compiler not found and could not be installed via package manager; please install Rust manually (https://rustup.rs) and re-run."
    return 1
}

install_python_deps() {
    echo "Upgrading pip ..."
    pip install --upgrade pip

    echo "Installing Python dependencies from requirements.txt ..."
    pip install -r requirements.txt
}

main() {
    check_python

    install_system_deps
    setup_venv

    ensure_rust

    install_python_deps

    echo
    echo "Environment setup complete."
    echo "To use it in a new shell, run:"
    echo "  source .venv/bin/activate"
}

main "$@"
