# shellcheck shell=bash
#
# Tolerant .env loader: handles VAR=value, VAR = "value", VAR='value', and
# any whitespace around the '='. Sourced by the Makefile targets and the
# launchers.
#
# Usage (from another sourced script, with bash -u tolerated):
#   source scripts/shared/_load_env.sh
#   _load_env .env
#
# Exports every parsed assignment. Comments (#...) and blank lines are
# skipped. Lines that don't match KEY=VALUE format are also skipped silently
# (mirrors python-dotenv's permissive behavior).
_load_env() {
  local file="$1"
  [[ -f "$file" ]] || return 0
  local line key val
  while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip comments and blank lines
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue
    # Match KEY [whitespace] = [whitespace] VALUE
    if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"
      val="${BASH_REMATCH[2]}"
      # Trim trailing whitespace
      val="${val%"${val##*[![:space:]]}"}"
      # Strip surrounding double or single quotes (one pair)
      if [[ "$val" =~ ^\"(.*)\"$ ]] || [[ "$val" =~ ^\'(.*)\'$ ]]; then
        val="${BASH_REMATCH[1]}"
      fi
      export "$key=$val"
    fi
  done < "$file"
}
