#!/bin/bash
# env.sh: Setup environment for Q3 Graph Indexing

echo "Initializing environment setup..."

install_pkg() {
    PKG_NAME=$1
    IMPORT_NAME=$2
    if ! python3 -c "import $IMPORT_NAME" &> /dev/null; then
        echo "Package '$PKG_NAME' not found. Installing..."
        pip install "$PKG_NAME"
        if [ $? -ne 0 ]; then
             echo "Failed to install $PKG_NAME. Attempting user install..."
             pip install --user "$PKG_NAME"
        fi
    else
        echo "Package '$PKG_NAME' is already installed."
    fi
}

install_pkg "gspan-mining" "gspan_mining"
install_pkg "networkx" "networkx"
install_pkg "numpy" "numpy"
install_pkg "pandas" "pandas"
install_pkg "joblib" "joblib"

echo "Environment setup completed successfully."
