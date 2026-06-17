MODEL_DIR="./resources/models/cross-encoder"
mkdir -p $MODEL_DIR

if [ ! -d "$MODEL_DIR" ]; then
  echo "Downloading model to $MODEL_DIR..."
  hf download cross-encoder/ms-marco-MiniLM-L-6-v2 --local-dir "$MODEL_DIR"
else
  echo "Model already exists at $MODEL_DIR, skipping download."
fi
