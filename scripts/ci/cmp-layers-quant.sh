#!/bin/bash
#
# Compares the outputs of tests captured with `--features capture-layers-quant`
# that get stored in `layers-quant` dir.
#
# It expects two directories in the working dir:
# - `layers-quant-base` - a baseline for comparison (typically from base branch)
# - `layers-quant` - outputs on the HEAD
#
# Note that the layers and inputs that are no longer present in the HEAD will be
# logged and ignored. If any outputs differ, the script will print their diffs
# and exit with non-zero code.

set -Eo pipefail

result=0

for layer in layers-quant-base/*/
do
  layer="${layer%/}" # strip trailing slash
  layer="${layer##*/}" # strip path and leading slash
  if [ -d "layers-quant/$layer" ]; then

    for input in layers-quant-base/"$layer"/*/
    do
      input="${input%/}" # strip trailing slash
      input="${input##*/}" # strip path and leading slash

      base="layers-quant-base/$layer/$input"
      head="layers-quant/$layer/$input"
      if [ -d "$head" ]; then
        # Compare output hashes
        if ! cmp --silent -- "$base/output_hash.txt" "$head/output_hash.txt"; then
          result=1
          echo "Output for input \"$layer/$input\" has changed:"
          echo "Base: $base/data.json"
          echo "Head: $head/data.json"
          diff "$base/data.json" "$head/data.json"
        fi
      else
        echo "Input \"$layer/$input\" is no longer present"
        result=1
      fi
    done
  else
    echo "Layer \"$layer\" is no longer present"
  fi
done

exit $result
