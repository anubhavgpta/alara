"""PromptBuilder for Gemini-based intent classification.

The prompt is constructed *entirely* from ``ACTION_REGISTRY`` and tailored to
the current utterance via semantic example selection. No action names or
schemas are hardcoded here.
"""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from .action_registry import ACTION_REGISTRY, ActionDefinition, ActionRegistry


class PromptBuilder:
    """Builds Gemini system prompts from the action registry."""

    def __init__(
        self,
        registry: ActionRegistry | None = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.registry: ActionRegistry = registry or ACTION_REGISTRY
        self.embedding_model_name = embedding_model_name

        logger.info(
            "Initializing PromptBuilder with {} actions using embedding model '{}'",
            len(self.registry.actions_by_name),
            self.embedding_model_name,
        )

        self._model = SentenceTransformer(self.embedding_model_name)

        # Global index of all examples for fast similarity search:
        # (action_name, example_index, embedding_vector)
        self._example_index: List[Tuple[str, int, np.ndarray]] = []
        # Tag → list of (action_name, example_index)
        self._tag_to_examples: Dict[str, List[Tuple[str, int]]] = {}

        self._precompute_embeddings()

        @lru_cache(maxsize=512)
        def _cached_build(utterance: str) -> str:
            return self._build_prompt(utterance)

        # Bound LRU cache closure; used by build().
        self._cached_build = _cached_build

    # Public API -------------------------------------------------------------

    def build(self, utterance: str) -> str:
        """Return a full system prompt tailored to the input utterance.

        The prompt includes:
        - grouped action list with descriptions and param schemas
        - selected few-shot examples (semantic + coverage)
        - instructions for phonetic inference, compound commands, and
          confidence calibration
        """
        return self._cached_build(utterance.strip())

    # Internal helpers -------------------------------------------------------

    def _precompute_embeddings(self) -> None:
        """Precompute embeddings for all examples at startup."""
        texts: List[str] = []
        meta: List[Tuple[str, int]] = []

        for action in self.registry.all_actions():
            for idx, (utterance, _expected) in enumerate(action.examples):
                texts.append(utterance)
                meta.append((action.name, idx))
                for tag in action.tags:
                    self._tag_to_examples.setdefault(tag, []).append((action.name, idx))

        if not texts:
            logger.warning("PromptBuilder initialized with no examples in registry.")
            return

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        self._example_index = [
            (action_name, ex_idx, vec)
            for (action_name, ex_idx), vec in zip(meta, embeddings)
        ]

        total_examples = len(self._example_index)
        logger.info(
            "PromptBuilder precomputed {} example embeddings across {} actions",
            total_examples,
            len(self.registry.actions_by_name),
        )

    def _select_examples(
        self, utterance: str, top_k: int = 2
    ) -> List[Tuple[ActionDefinition, str, Dict]]:
        """Select few-shot examples based on semantic similarity + tag coverage."""
        if not self._example_index:
            return []

        query_vec = self._model.encode(
            [utterance], convert_to_numpy=True, normalize_embeddings=True
        )[0]

        sims: List[Tuple[float, str, int]] = []
        for action_name, ex_idx, vec in self._example_index:
            sim = float(np.dot(query_vec, vec))
            sims.append((sim, action_name, ex_idx))

        sims.sort(key=lambda t: t[0], reverse=True)

        selected_keys: set[Tuple[str, int]] = set()
        for _sim, action_name, ex_idx in sims[:top_k]:
            selected_keys.add((action_name, ex_idx))

        # Add one random example per tag group for broader coverage.
        rng = random.Random(utterance)  # deterministic per utterance
        for tag, pairs in self._tag_to_examples.items():
            remaining = [p for p in pairs if p not in selected_keys]
            if not remaining:
                continue
            chosen = rng.choice(remaining)
            selected_keys.add(chosen)

        selected: List[Tuple[ActionDefinition, str, Dict]] = []
        for action_name, ex_idx in selected_keys:
            action = self.registry.get(action_name)
            if not action:
                continue
            if ex_idx >= len(action.examples):
                continue
            utter, expected = action.examples[ex_idx]
            selected.append((action, utter, expected))

        # Stable ordering: by action name then utterance for deterministic prompts.
        selected.sort(key=lambda t: (t[0].name, t[1]))
        return selected

    def _render_actions_by_tag(self) -> str:
        """Render allowed actions grouped by tag."""
        lines: List[str] = []
        for tag in sorted(self.registry.tag_to_actions.keys()):
            lines.append(f"Tag group: {tag}")
            for action_name in self.registry.tag_to_actions[tag]:
                action = self.registry.get(action_name)
                if not action:
                    continue
                lines.append(f"- {action.name}: {action.description}")
                if action.params_schema:
                    lines.append(f"  Params JSON schema: {action.params_schema}")
            lines.append("")  # blank line between groups
        return "\n".join(lines).rstrip()

    def _render_examples(
        self, examples: Sequence[Tuple[ActionDefinition, str, Dict]]
    ) -> str:
        """Render few-shot examples section."""
        if not examples:
            return "No examples available."

        blocks: List[str] = []
        for action, utterance, expected in examples:
            blocks.append(
                f"User: \"{utterance}\"\n"
                f"Assistant JSON:\n"
                f"{expected}"
            )
        return "\n\n".join(blocks)

    def _build_prompt(self, utterance: str) -> str:
        """Internal, uncached prompt construction."""
        actions_section = self._render_actions_by_tag()
        examples = self._select_examples(utterance)
        examples_section = self._render_examples(examples)

        prompt = f"""
You are the intent classification engine for the Windows voice assistant "Alara".

Your job is to convert noisy speech-to-text transcripts into a **single best action**
for the assistant to execute, using the ACTION_REGISTRY defined below as the
**single source of truth**. You MUST only use actions and parameters described
in this registry.

Speech recognition can be imperfect. You MUST:
- Infer intent **phonetically** when needed (e.g. "krome" → "chrome").
- Treat minor misspellings, homophones, or STT glitches as noise, not as new words.
- Prefer mapping to a real action over returning ``"unknown"``.
- Only use ``"unknown"`` when the user truly expresses confusion or no mapping is possible.

COMPOUND COMMANDS:
- Users may issue compound commands like "open Chrome and go to GitHub".
- If you can represent the intent as a single action, do so.
- If multiple actions are clearly required, you may return a **list** of action
  objects instead of a single object.

CONFIDENCE CALIBRATION:
- Always include a numeric ``confidence`` between 0.0 and 1.0.
- Use >= 0.8 only for very clear, unambiguous matches.
- Use around 0.5 when you are unsure between multiple plausible actions.
- Use <= 0.3 only when you are forced into ``\"unknown\"`` or very low certainty.

OUTPUT FORMAT (STRICT JSON):
- Prefer a **single JSON object** with:
  - ``"action"``: string, one of the registry action names or an alias that maps to one.
  - ``"params"``: object of parameters.
  - ``"confidence"``: float between 0.0 and 1.0.
- If the intent truly requires multiple steps, you may instead return a JSON
  **array of such objects**.
- Do NOT include any extra commentary, explanations, or markdown. Return ONLY JSON.

ALLOWED ACTIONS (GROUPED BY TAG):
{actions_section}

FEW-SHOT EXAMPLES (FOR GUIDANCE):
{examples_section}

CURRENT USER UTTERANCE:
"{utterance}"

Now reason carefully about the utterance and return the JSON response.
"""
        return prompt.strip()

