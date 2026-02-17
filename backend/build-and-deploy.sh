#!/bin/bash

#########################################################
#                1. Build Stage
########################################################

# Pre-build: ensure clean state
rm -rf __pycache__ .pytest_cache
find app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

#########################################################
#                2. Deploy Stage
########################################################

SERVICE_NAME="matching-service"
REGION="us-central1"
MEMORY="1Gi"
TIMEOUT="300s"

PROJECT_ID=$1
echo "Using project ID: $PROJECT_ID"

ENV_FILE_PATH=$2
echo "Using environment file: $ENV_FILE_PATH"

gcloud run deploy $SERVICE_NAME \
  --project="$PROJECT_ID" \
  --source=. \
  --region=$REGION \
  --memory=$MEMORY \
  --timeout=$TIMEOUT \
  --no-allow-unauthenticated \
  --env-vars-file="$ENV_FILE_PATH"
