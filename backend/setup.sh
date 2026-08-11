# Vendor one cross-encoder checkpoint per language, each in its own directory named after the
# HF repo. app/services/cross_encoder/reranker.py looks the checkpoint up by that name, so the
# layout here must stay `$MODEL_DIR/<repo-name>` (see _vendored_model_dirs).
MODEL_DIR="./resources/models/cross-encoder"

# cross-encoder/<repo>: en -> ms-marco-MiniLM-L-6-v2, es -> mmarco-mMiniLMv2-L12-H384-v1
MODELS="ms-marco-MiniLM-L-6-v2 mmarco-mMiniLMv2-L12-H384-v1"

for model in $MODELS; do
  target="$MODEL_DIR/$model"
  # config.json, not the directory: a half-finished download leaves the directory behind.
  if [ -f "$target/config.json" ]; then
    echo "Model already exists at $target, skipping download."
  else
    echo "Downloading cross-encoder/$model to $target..."
    hf download "cross-encoder/$model" --local-dir "$target"
  fi
done
