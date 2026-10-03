#!/usr/bin/env bash
# Fetches every PyPI wheel of the PJRT CPU plugin, extracts the library from
# each and produces one archive per platform (pjrt-cpu-<os>-<arch>.{tar.gz,zip}).
#
# Optional variables:
#   PJRT_PKG      PyPI package name          (default: xla-cpu-pjrt)
#   PJRT_VERSION  exact version              (default: latest)
#   OUT_DIR       output directory           (default: dist/pjrt)
#   RELEASE_TAG   if set: publishes the archives to that GitHub release (via gh)
#   REPO          owner/repo, for the printed zig fetch commands
set -euo pipefail

PKG="${PJRT_PKG:-xla-cpu-pjrt}"
VERSION="${PJRT_VERSION:-}"
OUT_DIR="${OUT_DIR:-dist/pjrt}"
RELEASE_TAG="${RELEASE_TAG:-}"
REPO="${REPO:-OWNER/REPO}"

for cmd in curl jq unzip tar; do
	command -v "$cmd" >/dev/null || {
		echo "ERROR: '$cmd' is required" >&2
		exit 1
	}
done

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi; }

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
: >"$OUT_DIR/SHA256SUMS"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

if [ -n "$VERSION" ]; then
	api="https://pypi.org/pypi/$PKG/$VERSION/json"
else
	api="https://pypi.org/pypi/$PKG/json"
fi
curl -fsSL "$api" -o "$work/meta.json"
ver="$(jq -r '.info.version' "$work/meta.json")"
echo ">> $PKG $ver"

assets=()
found=0

while IFS=$'\t' read -r fname url sha; do
	tag="${fname%.whl}"
	tag="${tag##*-}" # last field of the wheel name = platform tag

	case "$tag" in
	*musllinux*)
		echo "-- skipped (musl): $fname" >&2
		continue
		;;
	*manylinux* | linux_*)
		os=linux
		lib=libpjrt_cpu.so
		;;
	macosx*)
		os=macos
		lib=libpjrt_cpu.dylib
		;;
	win*)
		os=windows
		lib=pjrt_cpu.dll
		;;
	*)
		echo "-- skipped (unknown platform): $fname" >&2
		continue
		;;
	esac

	case "$tag" in
	*x86_64* | *amd64*) arch=x86_64 ;;
	*aarch64* | *arm64*) arch=aarch64 ;;
	*)
		echo "-- skipped (unknown arch or universal2): $fname" >&2
		continue
		;;
	esac

	echo ">> $fname  ->  $os/$arch"
	curl -fsSL "$url" -o "$work/$fname"

	got="$(sha256 "$work/$fname" | cut -d' ' -f1)"
	if [ "$got" != "$sha" ]; then
		echo "ERROR: SHA256 differs for $fname" >&2
		exit 1
	fi

	d="$work/unz-$os-$arch"
	mkdir -p "$d"
	unzip -q "$work/$fname" -d "$d"

	# Prefer a file whose name contains pjrt or xla, otherwise take the first one.
	src=""
	while IFS= read -r f; do
		if [ -z "$src" ]; then src="$f"; fi
		case "$(basename "$f")" in *pjrt* | *xla*)
			src="$f"
			break
			;;
		esac
	done < <(find "$d" -type f \( -name '*.so' -o -name '*.dylib' -o -name '*.dll' \) | sort)

	if [ -z "$src" ]; then
		echo "ERROR: no library in $fname" >&2
		exit 1
	fi
	echo "   library kept: ${src#"$d/"}"

	stage="$work/stage-$os-$arch"
	mkdir -p "$stage"
	cp "$src" "$stage/$lib"

	# Independent architecture check: the symbol name must show up.
	if ! grep -aq GetPjrtApi "$stage/$lib"; then
		echo "WARNING: GetPjrtApi not found in $os/$arch (wrong file?)" >&2
	fi

	base="pjrt-cpu-$os-$arch"
	if [ "$os" = windows ]; then
		command -v zip >/dev/null || {
			echo "ERROR: 'zip' is required for Windows" >&2
			exit 1
		}
		archive="$OUT_DIR/$base.zip"
		rm -f "$archive"
		(cd "$stage" && zip -q "$archive" "$lib")
	else
		archive="$OUT_DIR/$base.tar.gz"
		tar czf "$archive" -C "$stage" "$lib"
	fi

	(cd "$OUT_DIR" && sha256 "$(basename "$archive")") >>"$OUT_DIR/SHA256SUMS"
	assets+=("$archive")
	found=$((found + 1))
done < <(jq -r '.urls[] | select(.packagetype=="bdist_wheel") | [.filename, .url, .digests.sha256] | @tsv' "$work/meta.json")

if [ "$found" -eq 0 ]; then
	echo "ERROR: no usable wheel for $PKG $ver" >&2
	exit 1
fi

echo
echo ">> $found archive(s) in $OUT_DIR:"
ls -lh "$OUT_DIR"

if [ -n "$RELEASE_TAG" ]; then
	command -v gh >/dev/null || {
		echo "ERROR: 'gh' is required to publish" >&2
		exit 1
	}
	if ! gh release view "$RELEASE_TAG" >/dev/null 2>&1; then
		gh release create "$RELEASE_TAG" --title "$RELEASE_TAG" \
			--notes "PJRT CPU plugins extracted from $PKG $ver"
	fi
	gh release upload "$RELEASE_TAG" "${assets[@]}" "$OUT_DIR/SHA256SUMS"
	echo
	echo ">> For each archive, in kindred/:"
	for a in "${assets[@]}"; do
		n="$(basename "$a")"
		dep="$(echo "${n%.tar.gz}" | sed 's/\.zip$//; s/-/_/g')"
		echo "zig fetch --save=$dep https://github.com/$REPO/releases/download/$RELEASE_TAG/$n"
	done
	echo "(then add .lazy = true to each entry in build.zig.zon)"
fi