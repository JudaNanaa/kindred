#!/usr/bin/env bash
# Récupère le plugin PJRT CPU depuis PyPI et le copie dans third_party/pjrt/plugins/.
# Usage : scripts/fetch_pjrt.sh
# Variables optionnelles : PJRT_PKG (nom du paquet), PJRT_VERSION (version exacte)
set -euo pipefail

PKG="${PJRT_PKG:-xla-cpu-pjrt}"
VERSION="${PJRT_VERSION:-}" # vide = dernière version
DEST="third_party/pjrt/plugins"
OUT="$DEST/xla_cpu_pjrt.so"

mkdir -p "$DEST"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

spec="$PKG${VERSION:+==$VERSION}"
echo ">> Téléchargement de $spec"
python3 -m pip download --no-deps --only-binary=:all: -d "$tmp/wheel" "$spec"

whl="$(ls "$tmp"/wheel/*.whl | head -n1)"
echo ">> Roue : $(basename "$whl")"
unzip -q "$whl" -d "$tmp/unz"

mapfile -t libs < <(find "$tmp/unz" -name '*.so' | sort)
if [ "${#libs[@]}" -eq 0 ]; then
	echo "ERREUR : aucun .so trouvé dans la roue" >&2
	exit 1
fi
echo ">> Bibliothèques trouvées :"
printf '   %s\n' "${libs[@]#"$tmp/unz/"}"

# Préfère un fichier dont le nom contient pjrt ou xla, sinon prend le premier.
so="${libs[0]}"
for f in "${libs[@]}"; do
	case "$(basename "$f")" in *pjrt* | *xla*)
		so="$f"
		break
		;;
	esac
done

cp "$so" "$OUT"
echo ">> Copié : $OUT"
sha256sum "$OUT"

if command -v nm >/dev/null && nm -D "$OUT" 2>/dev/null | grep GetPjrtApi >/dev/null; then
	echo ">> OK : GetPjrtApi est exporté"
else
	echo "ATTENTION : GetPjrtApi introuvable dans $OUT (nm absent ou mauvais fichier ?)" >&2
fi
