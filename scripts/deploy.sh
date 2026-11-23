#!/usr/bin/env bash
# Deploy the RAG stack on this host: fetch secrets, pull the tested image,
# start the services and wait until the API reports ready.
#
#   scripts/deploy.sh                    deploy the latest image published by CI
#   scripts/deploy.sh <commit>           deploy (or roll back to) a specific commit
#   SMOKE=1 scripts/deploy.sh [commit]   also ask one question end to end
set -euo pipefail

PROFILE="${PROFILE:-cpu}"
REGION="${AWS_REGION:-eu-north-1}"
SMOKE="${SMOKE:-0}"
export APP_TAG="${1:-latest}"

cd "$(dirname "$0")/.."

show_logs_and_fail() {
  echo "$1" >&2
  docker compose --profile "$PROFILE" logs app --tail 40 >&2
  exit 1
}

# Secrets live in SSM Parameter Store; the instance role can read them.
get_param() {
  aws ssm get-parameter --region "$REGION" --name "/rag-llmops/langfuse/$1" \
    --with-decryption --query Parameter.Value --output text
}
export LANGFUSE_PUBLIC_KEY="$(get_param public_key)"
export LANGFUSE_SECRET_KEY="$(get_param secret_key)"
export LANGFUSE_BASE_URL="$(get_param base_url)"

# compose.yaml and configs come from the repository; the app comes as an image.
git pull --ff-only
docker compose --profile "$PROFILE" pull
docker compose --profile "$PROFILE" up -d --no-build

# Readiness: the vector index is available.
echo "Waiting for the API to be ready (image tag: $APP_TAG)..."
ready=false
for _ in $(seq 1 60); do
  if curl -fsS http://localhost:8080/ready >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 5
done
"$ready" || show_logs_and_fail "Not ready after 5 minutes. Has the ingestion job run?"
echo "Ready."

# Optional smoke test: one real question through embedding, retrieval and
# generation. Off by default to keep boot fast; run with SMOKE=1 after changes.
if [ "$SMOKE" = "1" ]; then
  echo "Smoke test: asking one question..."
  response="$(curl -fsS --max-time 180 -X POST http://localhost:8080/query \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the purpose of this Regulation?"}')" \
    || show_logs_and_fail "Smoke test failed: the API did not answer."

  python3 -c '
import json, sys
body = json.load(sys.stdin)
assert body["answer"].strip(), "empty answer"
assert body["sources"], "no sources"
' <<<"$response" || show_logs_and_fail "Smoke test failed: empty answer or no sources."
  echo "Smoke test passed."
fi
