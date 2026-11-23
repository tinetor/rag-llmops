#!/usr/bin/env bash
# Deploy the RAG stack on this host: fetch secrets, pull the tested image,
# start the services and wait until the API reports ready.
#
#   scripts/deploy.sh             deploy the latest image published by CI
#   scripts/deploy.sh <commit>    deploy (or roll back to) a specific commit
set -euo pipefail

PROFILE="${PROFILE:-cpu}"
REGION="${AWS_REGION:-eu-north-1}"
export APP_TAG="${1:-latest}"

cd "$(dirname "$0")/.."

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

echo "Waiting for the API to be ready (image tag: $APP_TAG)..."
for _ in $(seq 1 60); do
  if curl -fsS http://localhost:8080/ready >/dev/null 2>&1; then
    echo "Ready."
    exit 0
  fi
  sleep 5
done

echo "The API was not ready after 5 minutes. Has the ingestion job run?" >&2
docker compose --profile "$PROFILE" logs app --tail 40 >&2
exit 1
