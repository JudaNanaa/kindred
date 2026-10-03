#!/usr/bin/env bash
# Fetches the PJRT CPU plugin from PyPI and copies it into third_party/pjrt/plugins/.
# Usage: scripts/fetch_pjrt.sh
# Optional variables: PJRT_PKG (package name), PJRT_VERSION (exact version)
set -euo pipefail

PKG="${PJRT_PKG:-xla-cpu-pjrt}"
VERSION="${PJRT_VERSION:-}" # empty = latest version
DEST="third_party/pjrt/plugins"
OUT="$DEST/xla_cpu_pjrt.so"

mkdir -p "$DEST"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

spec="$PKG${VERSION:+==$VERSION}"
echo ">> Downloading $spec"
python3 -m pip download --no-deps --only-binary=:all: -d "$tmp/wheel" "$spec"

whl="$(ls "$tmp"/wheel/*.whl | head -n1)"
echo ">> Wheel: $(basename "$whl")"
unzip -q "$whl" -d "$tmp/unz"

mapfile -t libs < <(find "$tmp/unz" -name '*.so' | sort)
if [ "${#libs[@]}" -eq 0 ]; then
	echo "ERROR: no .so found in the wheel" >&2
	exit 1
fi
echo ">> Libraries found:"
printf '   %s\n' "${libs[@]#"$tmp/unz/"}"

# Prefer a file whose name contains pjrt or xla, otherwise take the first one.
so="${libs[0]}"
for f in "${libs[@]}"; do
	case "$(basename "$f")" in *pjrt* | *xla*)
		so="$f"
		break
		;;
	esac
done

cp "$so" "$OUT"
echo ">> Copied: $OUT"
sha256sum "$OUT"

if command -v nm >/dev/null && nm -D "$OUT" 2>/dev/null | grep GetPjrtApi >/dev/null; then
	echo ">> OK: GetPjrtApi is exported"
else
	echo "WARNING: GetPjrtApi not found in $OUT (nm missing or wrong file?)" >&2
fi