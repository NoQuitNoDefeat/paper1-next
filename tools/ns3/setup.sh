#!/usr/bin/env bash
# Build this project's own ns-3 tree (ns-3 + ns3-ai + fanet-scheduler contrib).
#
#   tools/ns3/setup.sh                 # clone (if needed), pin, patch, configure, build
#   NS3_SOURCE=/path/to/ns-3-dev NS3_AI_SOURCE=/path/to/ns3-ai tools/ns3/setup.sh
#
# Sources default to the upstream repositories in dependencies/*.lock; local
# mirrors can be given instead.  The tree lives in simulator/ns-3-dev (not versioned);
# the contrib module is simulator/ns-3-external-contrib/fanet-scheduler (versioned).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOCKS="$ROOT/dependencies"
NS3_DIR="$ROOT/simulator/ns-3-dev"
lock() { grep "^$2=" "$LOCKS/$1" | cut -d= -f2-; }

NS3_COMMIT="$(lock ns3.lock commit)"
AI_COMMIT="$(lock ns3-ai.lock commit)"
NS3_SOURCE="${NS3_SOURCE:-$(lock ns3.lock repository)}"
NS3_AI_SOURCE="${NS3_AI_SOURCE:-$(lock ns3-ai.lock repository)}"

if [ ! -d "$NS3_DIR/.git" ]; then git clone --quiet "$NS3_SOURCE" "$NS3_DIR"; fi
git -C "$NS3_DIR" -c advice.detachedHead=false checkout --quiet "$NS3_COMMIT"
[ "$(git -C "$NS3_DIR" rev-parse HEAD)" = "$NS3_COMMIT" ] || { echo "ns-3 commit mismatch"; exit 1; }

AI_DIR="$NS3_DIR/contrib/ai"
if [ ! -d "$AI_DIR/.git" ]; then git clone --quiet "$NS3_AI_SOURCE" "$AI_DIR"; fi
if [ -z "$(git -C "$AI_DIR" status --porcelain)" ]; then
  git -C "$AI_DIR" -c advice.detachedHead=false checkout --quiet "$AI_COMMIT"
  IFS=',' read -ra PATCHES <<< "$(lock ns3-ai.lock patches)"
  for p in "${PATCHES[@]}"; do git -C "$AI_DIR" apply "$LOCKS/$p"; echo "applied $p"; done
else
  echo "contrib/ai already patched (working tree modified); leaving as is"
fi
[ "$(git -C "$AI_DIR" rev-parse HEAD)" = "$AI_COMMIT" ] || { echo "ns3-ai commit mismatch"; exit 1; }

cmake -S "$NS3_DIR" -B "$NS3_DIR/cmake-cache" -G Ninja \
  -DCMAKE_BUILD_TYPE=default -DNS3_EXAMPLES=ON -DNS3_TESTS=ON -DNS3_PYTHON_BINDINGS=OFF -DNS3_WARNINGS_AS_ERRORS=OFF \
  -DPython_EXECUTABLE="$ROOT/.venv/bin/python" > "$NS3_DIR/configure.log"
echo "configured (log: simulator/ns-3-dev/configure.log)"

TARGETS="fanet-bridge fanet_bridge_native fanet-bridge-tests fanet-controlled fanet-controlled-tests
         fanet-scheduled-radio-tests fanet-moving-channel-tests fanet-control-channel-tests"
[ "${NS3_SKIP_BUILD:-0}" = 1 ] || cmake --build "$NS3_DIR/cmake-cache" --target $TARGETS -j "${JOBS:-6}"
echo "ns-3 build ready"
