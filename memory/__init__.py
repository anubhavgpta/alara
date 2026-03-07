"""Memory layer modules for session tracking, preferences, and reusable skills."""

from __future__ import annotations

import threading
import time
from typing import Any

from loguru import logger

from alara.core.orchestrator import OrchestratorResult
from alara.memory.database import DatabaseManager
from alara.memory.models import MemoryContext
from alara.memory.preferences import PreferenceMemory
from alara.memory.session import SessionMemory
from alara.memory.skills import SkillMemory
from alara.schemas.goal import GoalContext
from alara.schemas.task_graph import TaskGraph


class MemoryManager:
    """The single public interface to the entire memory layer."""
    
    _instance: MemoryManager | None = None
    _lock: threading.Lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> MemoryManager:
        """Get the singleton MemoryManager instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self) -> None:
        """Initialize the memory manager."""
        self.session = SessionMemory()
        self.preferences = PreferenceMemory()
        self.skills = SkillMemory()
        self.db = DatabaseManager.get_instance()
        logger.info("MemoryManager initialized")
    
    def build_context(self, goal: str, goal_context: GoalContext) -> MemoryContext:
        """
        Build a MemoryContext for the Planner to use.
        
        Args:
            goal: The goal string
            goal_context: The parsed goal context
            
        Returns:
            MemoryContext with relevant memory information
        """
        # Get recent goals
        recent_goals = self.session.get_recent(limit=5)
        
        # Get relevant skills
        relevant_skills = self.skills.search(goal, limit=3)
        
        # Get relevant preferences (path and tool categories)
        path_preferences = self.preferences.get_by_category("path")
        tool_preferences = self.preferences.get_by_category("tool")
        relevant_preferences = path_preferences + tool_preferences
        
        # Get known paths
        known_paths = self.preferences.get_all_path_aliases()
        
        # Build summary string for Gemini injection
        summary_parts = ["MEMORY CONTEXT:\n"]
        
        # Add known path aliases
        if known_paths:
            summary_parts.append("Known path aliases:\n")
            for alias, path in known_paths.items():
                summary_parts.append(f"  {alias} -> {path}\n")
            summary_parts.append("\n")
        
        # Add last executed paths from successful goals (last 3 only)
        if recent_goals:
            summary_parts.append("Last executed paths:\n")
            for entry in recent_goals[:3]:  # Only last 3 entries
                if entry.status == "success" and entry.execution_log:
                    for log_entry in entry.execution_log:
                        if log_entry.get("verified") and log_entry.get("verification_detail"):
                            detail = log_entry["verification_detail"]
                            if detail.startswith("Path exists: "):
                                extracted_path = detail[12:]  # Remove "Path exists: " prefix
                                summary_parts.append(f"  {entry.goal[:50]} → {extracted_path}\n")
                                break  # Only take first path from each successful entry
            summary_parts.append("\n")
        
        # Add recently completed goals
        if recent_goals:
            summary_parts.append("Recently completed goals:\n")
            for entry in recent_goals:
                summary_parts.append(f"  - {entry.goal} ({entry.status})\n")
            summary_parts.append("\n")
        
        # Add relevant skills
        if relevant_skills:
            summary_parts.append("Relevant skills available:\n")
            for skill in relevant_skills:
                summary_parts.append(
                    f"  - {skill.name}: {skill.goal_pattern} "
                    f"({skill.success_count} successful uses)\n"
                )
                summary_parts.append(f"    Steps: {len(skill.steps)} steps\n")
            summary_parts.append("\n")
        
        # Add user preferences
        if relevant_preferences:
            summary_parts.append("User preferences:\n")
            for pref in relevant_preferences:
                try:
                    import json
                    value = json.loads(pref.value)
                    summary_parts.append(f"  {pref.key}: {value}\n")
                except:
                    summary_parts.append(f"  {pref.key}: {pref.value}\n")
        
        # Add skill reference instruction if relevant skills exist
        if relevant_skills:
            summary_parts.append(
                "\nNOTE: A similar task has been completed successfully before. "
                "You may use the skill as a reference but adapt it to the "
                "current goal. Do not copy it blindly — verify it fits.\n"
            )
            
            # Add skill steps as reference
            for skill in relevant_skills[:1]:  # Only show the most relevant skill
                summary_parts.append(f"\nReference skill '{skill.name}':\n")
                for step in skill.steps:
                    summary_parts.append(
                        f"  {step['id']}. [{step['step_type']}] {step['operation']}: "
                        f"{step['description']}\n"
                    )
        
        summary = "".join(summary_parts)
        
        context = MemoryContext(
            session_id=self.session.session_id,
            recent_goals=recent_goals,
            relevant_skills=relevant_skills,
            relevant_preferences=relevant_preferences,
            known_paths=known_paths,
            summary=summary,
        )
        
        logger.debug("Built memory context with {} recent goals, {} skills, {} preferences",
                    len(recent_goals), len(relevant_skills), len(relevant_preferences))
        
        return context
    
    def after_execution(self, goal: str, goal_context: GoalContext,
                        task_graph: TaskGraph, result: OrchestratorResult,
                        entry_id: str, duration_ms: float) -> None:
        """
        Called after every goal execution completes.
        
        Args:
            goal: The original goal string
            goal_context: The parsed goal context
            task_graph: The executed task graph
            result: The execution result
            entry_id: The session entry ID
            duration_ms: Execution duration in milliseconds
        """
        try:
            # Complete the session entry
            self.session.complete_goal(entry_id, result)
            
            # If successful, store skill and infer preferences
            if result.success:
                # Store as skill if applicable
                skill = self.skills.store(goal, goal_context, task_graph, result, duration_ms)
                if skill:
                    logger.info("Stored skill: {}", skill.name)
                
                # Infer preferences from execution
                self.preferences.infer_from_execution(goal, task_graph, result)
            
            logger.info(
                "Memory update completed: goal={}, success={}, steps={}",
                goal[:50], result.success, result.steps_completed
            )
            
        except Exception as e:
            logger.warning("Memory update failed: {}", e)
    
    def health_check(self) -> dict[str, Any]:
        """
        Return combined health status of the memory system.
        
        Returns:
            Dictionary with health information
        """
        try:
            db_health = self.db.health_check()
            session_stats = self.session.get_stats()
            skill_stats = self.skills.get_stats()
            preference_export = self.preferences.export()
            
            return {
                "database": db_health,
                "session_id": self.session.session_id,
                "session_goals": len(self.session.get_current_session_entries()),
                "total_skills": skill_stats["total_skills"],
                "total_preferences": len(preference_export),
                "session_stats": session_stats,
                "skill_stats": skill_stats,
            }
            
        except Exception as e:
            logger.error("Memory health check failed: {}", e)
            return {
                "status": "error",
                "error": str(e),
            }


__all__ = [
    "MemoryManager",
    "MemoryContext",
    "SessionMemory",
    "PreferenceMemory", 
    "SkillMemory",
    "DatabaseManager",
]
