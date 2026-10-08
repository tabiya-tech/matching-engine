"""Hugging Face cross-encoder (sentence-transformers) client used for stage-2 reranking."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from app.config import CROSS_ENCODER_BATCH_SIZE, cross_encoder_model_name
from app.languages import CANONICAL_LANGUAGE, default_language, normalise_language

logger = logging.getLogger(__name__)


class ICrossEncoderClient(ABC):
    """Interface for scoring [query, passage] pairs with a cross-encoder."""

    model_name: str
    """The checkpoint (Hugging Face id) this client scores with."""

    language: str
    """The language whose checkpoint this client serves."""

    @abstractmethod
    def warmup(self) -> None:
        """
        Loads the model so the first scoring call does not pay the load.

        :raises ImportError: If sentence-transformers is not installed
        """
        raise NotImplementedError()

    @abstractmethod
    def predict_scores(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Scores each [query, passage] pair.

        :param pairs: The pairs to score
        :return: One relevance logit per pair, in input order
        """
        raise NotImplementedError()


class CrossEncoderClient(ICrossEncoderClient):
    """Lazy-loaded Hugging Face cross-encoder (sentence-transformers).

    One instance serves one language: the query and passage are skill-label text, so an
    English-only checkpoint scores Spanish labels poorly. ``language`` picks the checkpoint
    from that language's config (``CROSS_ENCODER_MODEL_NAME_<LANG>`` overrides it), and the
    vendored model directory it looks for first.
    """

    def __init__(
        self,
        *,
        batch_size: Optional[int] = None,
        language: Optional[str] = None,
    ) -> None:
        self.batch_size = int(batch_size or CROSS_ENCODER_BATCH_SIZE)
        self.language = normalise_language(language) if language else default_language()
        self.model_name = cross_encoder_model_name(self.language)
        self._model = None
        self._logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _is_checkpoint_dir(path: Path) -> bool:
        """A directory holding an actual checkpoint, not just a parent of several.

        ``resources/models/cross-encoder/`` is a parent in the current layout, so ``is_dir()``
        alone would hand sentence-transformers a directory with no weights in it.
        """
        return path.is_dir() and (path / "config.json").is_file()

    def _vendored_model_dirs(self) -> List[Path]:
        """Local checkpoint directories to try, most specific first.

        ``setup.sh`` vendors one directory *per checkpoint* under
        ``resources/models/cross-encoder/<repo-name>`` (``ms-marco-MiniLM-L-6-v2`` for
        English, ``mmarco-mMiniLMv2-L12-H384-v1`` for Spanish), so one image can serve every
        language. Two earlier layouts are still accepted so an existing image or mounted
        volume keeps working: ``cross-encoder-<lang>/``, and the flat ``cross-encoder/``
        directory that predates language support (canonical language only, and only when it
        really holds a checkpoint).
        """
        # backend/app/clients/cross_encoder_client.py -> parents[2] == backend/
        models = Path(__file__).resolve().parents[2] / "resources" / "models"
        repo = self.model_name.strip().strip("/")
        checkpoint = repo.rsplit("/", 1)[-1]
        dirs = [
            models / f"cross-encoder-{self.language}",
            # `models / repo` covers HF ids under the cross-encoder org, i.e. exactly the
            # `cross-encoder/<name>` paths setup.sh writes; the next entry covers ids from
            # any other org (a CROSS_ENCODER_MODEL_NAME_<LANG> override) vendored the same way.
            models / repo,
            models / "cross-encoder" / checkpoint,
        ]
        if self.language == CANONICAL_LANGUAGE:
            dirs.append(models / "cross-encoder")
        out: List[Path] = []
        for d in dirs:
            if d not in out:
                out.append(d)
        return out

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415 — heavy import
        except ImportError as e:
            raise ImportError(
                "Cross-encoder reranking requires sentence-transformers. "
                "Install backend dependencies: pip install sentence-transformers"
            ) from e
        for path_name in self._vendored_model_dirs():
            if not self._is_checkpoint_dir(path_name):
                continue
            try:
                logger.info(
                    "Loading cross-encoder model %s (language=%s)",
                    path_name,
                    self.language,
                )
                # sentence-transformers needs a str (it does `"\\" in name` / `name.count("/")`
                # checks); a Path raises "argument of type 'WindowsPath' is not iterable".
                self._model = CrossEncoder(str(path_name))
                return
            except Exception as e:
                logger.exception(e)
        logger.info(
            "Loading cross-encoder model %s (language=%s)",
            self.model_name,
            self.language,
        )
        self._model = CrossEncoder(self.model_name)

    def warmup(self) -> None:
        """Load the Hugging Face model once (can take tens of seconds on first use)."""

        self._ensure_model()

    def predict_scores(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Return relevance scores for each [query, passage] pair."""

        if not pairs:
            return []
        self._ensure_model()
        assert self._model is not None
        raw = self._model.predict(
            pairs,
            batch_size=max(1, self.batch_size),
            show_progress_bar=False,
        )
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim == 2:
            if arr.shape[1] == 1:
                arr = arr[:, 0]
            else:
                arr = arr[:, -1]
        elif arr.ndim > 2:
            arr = arr.reshape(-1)
        flat = arr.reshape(-1).tolist()
        out: List[float] = []
        for x in flat:
            try:
                out.append(round(float(x), 6))
            except (TypeError, ValueError):
                out.append(0.0)
        return out
