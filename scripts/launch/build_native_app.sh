#!/bin/zsh
# Build a local app bundle so macOS has a camera usage description and a stable app identity.
set -eu
cd "$(dirname "$0")/../.."
app="$PWD/.venv/Visual Tracker.app"
stamp="$PWD/.venv/visual-tracker-native.buildstamp"
if [[ -x "$app/Contents/MacOS/VisualTracker" && -f "$stamp" && "$stamp" -nt native/main.swift && "$stamp" -nt native/Info.plist ]]; then
  exit 0
fi
if ! xcrun --find swiftc >/dev/null 2>&1; then
  echo "Install Apple's command line tools first: xcode-select --install"
  exit 1
fi
mkdir -p "$app/Contents/MacOS" .venv/swift-module-cache
rm -f "$stamp" "$app/Contents/.build-complete"
cp native/Info.plist "$app/Contents/Info.plist"
xcrun swiftc -O -swift-version 5 -target arm64-apple-macosx26.0 \
  -module-cache-path "$PWD/.venv/swift-module-cache" \
  -framework AppKit -framework AVFoundation native/main.swift \
  -o "$app/Contents/MacOS/VisualTracker"
codesign --force --sign - "$app"
codesign --verify --deep --strict "$app"
touch "$stamp"
