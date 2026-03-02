"""Post-LLM action and parameter normalization.

All behavior is driven off the ACTION_REGISTRY. No action names are
hardcoded here; aliases, normalization rules, defaults, and cross-action
corrections all come from the registry.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Tuple

from loguru import logger

from .action_registry import ACTION_REGISTRY, ActionDefinition, ActionRegistry


class ActionNormalizer:
    """Normalizes raw LLM outputs into canonical actions + params."""

    def __init__(self, registry: ActionRegistry | None = None):
        self.registry: ActionRegistry = registry or ACTION_REGISTRY

        # Precompile regexes for all normalization rules for speed.
        self._compiled_rules: Dict[str, list[tuple[str, Any]]] = {}
        for action in self.registry.all_actions():
            compiled_for_action: list[tuple[str, Any]] = []
            for rule in action.normalization_rules:
                pattern = (
                    re.compile(rule.regex_extract, flags=re.IGNORECASE)
                    if rule.regex_extract
                    else None
                )
                compiled_for_action.append((rule.param, (rule, pattern)))
            self._compiled_rules[action.name] = compiled_for_action

    # Public API -------------------------------------------------------------

    def normalize(
        self, action_name: str, params: Dict[str, Any], transcript: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Normalize an action+params pair against the registry.

        Steps:
        1. Resolve action aliases to canonical names.
        2. Apply per-action param key aliases.
        3. Apply value mapping rules.
        4. Apply regex extraction rules for missing params.
        5. Inject default params.
        6. Apply cross-action correction rules.
        """
        transcript = transcript or ""
        original_action = action_name

        canonical_action = self.registry.resolve_action_name(str(action_name))
        definition = self.registry.get(canonical_action)
        if not definition:
            logger.debug(
                "Unknown action '{}' from LLM; mapping to 'unknown'", canonical_action
            )
            canonical_action = "unknown"
            definition = self.registry.get(canonical_action)

        normalized_params = copy.deepcopy(params or {})
        normalized_params = self._apply_param_aliases(definition, normalized_params)
        normalized_params = self._apply_value_mappings(definition, normalized_params)
        normalized_params = self._apply_regex_extracts(
            definition, normalized_params, transcript
        )
        normalized_params = self._inject_defaults(definition, normalized_params)

        corrected_action, corrected_params = self._apply_cross_action_rules(
            canonical_action, normalized_params, transcript
        )

        logger.bind(
            component="ActionNormalizer",
            original_action=original_action,
            canonical_action=canonical_action,
            corrected_action=corrected_action,
        ).debug("Normalization complete")

        return corrected_action, corrected_params

    # Internal helpers -------------------------------------------------------

    def _apply_param_aliases(
        self, definition: ActionDefinition, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not definition.param_aliases:
            return params

        alias_map = {k.lower(): v for k, v in definition.param_aliases.items()}
        normalized: Dict[str, Any] = {}
        for key, value in params.items():
            canonical_key = alias_map.get(str(key).lower(), key)
            normalized[canonical_key] = value
        return normalized

    def _apply_value_mappings(
        self, definition: ActionDefinition, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        compiled = self._compiled_rules.get(definition.name, [])
        if not compiled:
            return params

        for param_name, (rule, _pattern) in compiled:
            if rule.mapping and param_name in params:
                value = params.get(param_name)
                if isinstance(value, str):
                    lowered = value.lower().strip()
                    for src, dst in rule.mapping.items():
                        if lowered == src.lower():
                            params[param_name] = dst
                            break
        return params

    def _apply_regex_extracts(
        self,
        definition: ActionDefinition,
        params: Dict[str, Any],
        transcript: str,
    ) -> Dict[str, Any]:
        compiled = self._compiled_rules.get(definition.name, [])
        if not compiled:
            return params

        for param_name, (rule, pattern) in compiled:
            if not pattern or not rule.regex_extract:
                continue

            current = params.get(param_name)
            if current not in (None, "", [], {}):
                continue

            match = pattern.search(transcript)
            if not match:
                continue

            extracted = match.group(1) if match.groups() else match.group(0)
            params[param_name] = extracted

        return params

    def _inject_defaults(
        self, definition: ActionDefinition, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        for key, default_val in (definition.default_params or {}).items():
            if key not in params or params[key] in (None, ""):
                params[key] = default_val
        return params

    def _apply_cross_action_rules(
        self,
        action_name: str,
        params: Dict[str, Any],
        transcript: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Apply registry-wide cross-action correction rules."""
        text = transcript.lower()

        for rule in self.registry.cross_action_rules:
            if self.registry.resolve_action_name(action_name) != rule.source_action:
                continue

            if rule.transcript_regex:
                if not re.search(rule.transcript_regex, text, flags=re.IGNORECASE):
                    continue

            if rule.required_missing_params:
                if any(
                    p in params and params.get(p) not in (None, "", [], {})
                    for p in rule.required_missing_params
                ):
                    continue

            if rule.required_params:
                all_match = True
                for key, expected in rule.required_params.items():
                    if params.get(key) != expected:
                        all_match = False
                        break
                if not all_match:
                    continue

            target_def = self.registry.get(rule.target_action)
            if not target_def:
                continue

            new_params = copy.deepcopy(params)
            new_params.update(rule.set_params or {})

            # Re-run per-action normalization for the target action, but skip
            # cross-action rules to avoid loops.
            new_params = self._apply_param_aliases(target_def, new_params)
            new_params = self._apply_value_mappings(target_def, new_params)
            new_params = self._apply_regex_extracts(target_def, new_params, transcript)
            new_params = self._inject_defaults(target_def, new_params)

            logger.bind(
                component="ActionNormalizer",
                from_action=action_name,
                to_action=rule.target_action,
            ).debug("Applied cross-action correction rule")

            return rule.target_action, new_params

        return action_name, params

