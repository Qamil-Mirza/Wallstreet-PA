#!/bin/sh
set -eu

model="${OLLAMA_RESEARCH_MODEL:-llama3.1:8b}"
timeout="${OLLAMA_HEALTH_TIMEOUT_SECONDS:-60}"
export OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"

case "$model" in
    ''|*[!A-Za-z0-9._:/-]*)
        echo "ollama-init: invalid OLLAMA_RESEARCH_MODEL" >&2
        exit 2
        ;;
esac
if [ "${#model}" -gt 128 ] || ! printf '%s\n' "$model" | grep -Eq '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}(/[A-Za-z0-9][A-Za-z0-9._-]{0,63})*(:[A-Za-z0-9][A-Za-z0-9._-]{0,63})?$'; then
    echo "ollama-init: invalid OLLAMA_RESEARCH_MODEL" >&2
    exit 2
fi
case "$timeout" in
    ''|*[!0-9]*)
        echo "ollama-init: invalid OLLAMA_HEALTH_TIMEOUT_SECONDS" >&2
        exit 2
        ;;
esac
if ! printf '%s\n' "$timeout" | grep -Eq '^[1-9][0-9]{0,3}$'; then
    echo "ollama-init: invalid OLLAMA_HEALTH_TIMEOUT_SECONDS" >&2
    exit 2
fi

elapsed=0
while ! ollama list >/dev/null 2>&1; do
    if [ "$elapsed" -ge "$timeout" ]; then
        echo "ollama-init: Ollama did not become healthy" >&2
        exit 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
done

installed="$(ollama list)"
if printf '%s\n' "$installed" | awk 'NR > 1 {print $1}' | grep -Fx -- "$model" >/dev/null; then
    echo "ollama-init: model already present"
    exit 0
fi

ollama pull "$model"
