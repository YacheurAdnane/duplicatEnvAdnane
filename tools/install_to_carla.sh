#!/usr/bin/env bash
# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
# Copy a generated map into a CARLA source build and import it.
#   tools/install_to_carla.sh output/<name> [~/carla]
set -euo pipefail
OUT="${1:?usage: install_to_carla.sh output/<name> [carla_root]}"
CARLA="${2:-$HOME/carla}"
NAME="$(basename "$(realpath "$OUT")")"
SRC="$OUT/carla/$NAME"
[ -f "$SRC/$NAME.fbx" ] || { echo "no $SRC/$NAME.fbx"; exit 1; }
[ -d "$CARLA/Import" ] || { echo "no $CARLA/Import (need a CARLA source build)"; exit 1; }

others=$(find "$CARLA/Import" -mindepth 1 -maxdepth 1 -type d ! -name "$NAME" | wc -l)
if [ "$others" -gt 0 ]; then
  echo "Note: $CARLA/Import contains other packages; 'make import' imports all of them:"
  find "$CARLA/Import" -mindepth 1 -maxdepth 1 -type d ! -name "$NAME"
fi
rm -rf "$CARLA/Import/$NAME"
cp -r "$SRC" "$CARLA/Import/$NAME"
echo "Copied to $CARLA/Import/$NAME"
cd "$CARLA"
echo "Running: make import ARGS=\"--package=$NAME\"  (takes a while)"
make import ARGS="--package=$NAME"
echo
echo "Done. Next:"
echo "  make launch  -> Content/$NAME/Maps/$NAME  (open the map and press Play)"
echo "  optional: run $OUT/ue_semantic_tags.py in the editor for semantic tags"
echo "  make package ARGS=\"--packages=$NAME\"  to build a standalone package"
