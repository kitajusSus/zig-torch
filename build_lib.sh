#!/bin/bash
set -e

# Ustal rozszerzenie dla systemu operacyjnego
if [ "$(uname)" == "Darwin" ]; then
    # macOS
    EXT=".dylib"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Linux
    EXT=".so"
else
    # Windows
    EXT=".dll"
fi

# Stwórz katalogi jeśli nie istnieją
mkdir -p build
mkdir -p zigtorch

# Kompiluj bibliotekę
zig build-lib -dynamic -OReleaseFast -fPIC -lc -femit-bin=build/libnative${EXT} src/native.zig

# Kopiuj do katalogu pakietu
cp build/libnative${EXT} zigtorch/

echo "Zbudowano bibliotekę: zigtorch/libnative${EXT}"
