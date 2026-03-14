#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ ! -f "$DIR/forest_fire" ] || [ "$DIR/forest_fire.cpp" -nt "$DIR/forest_fire" ]; then
    g++ -O3 -std=c++11 "$DIR/forest_fire.cpp" -o "$DIR/forest_fire"
fi
"$DIR/forest_fire" "$1" "$2" "$3" "$4" "$5" "$6"
