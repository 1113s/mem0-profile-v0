import json
import logging
from typing import Any, Dict, List, Optional

from mem0.configs.category_prompts import MEMORY_CATEGORIZATION_PROMPT
from mem0.memory.category_storage import CategorySQLiteManager
from mem0.memory.utils import remove_code_blocks
from mem0.utils.factory import LlmFactory

logger = logging.getLogger(__name__)

BATCH_SIZE = 20


class CategoryManager:
    """Independent memory categorization manager, decoupled from Memory/AsyncMemory.

    Assigns category labels to memories via LLM and persists associations in SQLite.
    """

    def __init__(self, db_path: str, llm_config: dict):
        """
        Args:
            db_path: SQLite database path (can share the same file as history.db).
            llm_config: LLM configuration dict, e.g. {"provider": "openai", "config": {...}}.
        """
        self.category_db = CategorySQLiteManager(db_path)
        self.llm = LlmFactory.create(llm_config["provider"], llm_config.get("config", {}))
        self.category_db.seed_default_categories()

    # ----------------------------------------------------------------
    # Category CRUD (delegates to CategorySQLiteManager)
    # ----------------------------------------------------------------

    def list_categories(self, limit: int = 100, offset: int = 0) -> list:
        return self.category_db.list_categories(limit=limit, offset=offset)

    def create_category(self, name: str, description: Optional[str] = None) -> dict:
        return self.category_db.create_category(name, description)

    def delete_category(self, category_id: str) -> None:
        self.category_db.delete_category(category_id)

    def get_memory_categories(self, memory_id: str) -> List[str]:
        return self.category_db.get_categories_for_memory(memory_id)

    def get_memories_by_category(self, category_name: str, limit: int = 100, offset: int = 0) -> List[str]:
        return self.category_db.get_memories_by_category(category_name, limit=limit, offset=offset)

    def get_memory_ids_by_categories(self, category_names: List[str]) -> List[str]:
        return self.category_db.get_memory_ids_by_categories(category_names)

    def get_category_counts(self) -> List[Dict[str, Any]]:
        return self.category_db.get_category_counts()

    # ----------------------------------------------------------------
    # Core: Categorize memories
    # ----------------------------------------------------------------

    def categorize(self, memory_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Categorize memories returned by Memory.add().

        Args:
            memory_results: The "results" list from Memory.add(), each item has
                            {"id", "memory", "event"} (event is ADD/UPDATE/DELETE).

        Returns:
            List of {"memory_id": str, "categories": list[str]} for ADD/UPDATE items.
        """
        to_categorize = []
        for item in memory_results:
            event = item.get("event", "").upper()
            if event == "DELETE":
                self.category_db.remove_memory_categories(item["id"])
            elif event in ("ADD", "UPDATE"):
                to_categorize.append(item)

        if not to_categorize:
            return []

        # Process in batches
        all_results = []
        for i in range(0, len(to_categorize), BATCH_SIZE):
            batch = to_categorize[i : i + BATCH_SIZE]
            batch_results = self._categorize_batch(batch)
            all_results.extend(batch_results)

        return all_results

    def _categorize_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Categorize a batch of memory items via a single LLM call."""
        # Build category list for prompt
        categories = self.category_db.list_categories(limit=200)
        categories_text = "\n".join(f"- {c['name']}: {c['description'] or ''}" for c in categories)

        system_prompt = MEMORY_CATEGORIZATION_PROMPT.format(categories=categories_text)
        user_content = json.dumps(
            [{"memory_id": item["id"], "text": item["memory"]} for item in items],
            ensure_ascii=False,
        )

        try:
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
            )

            cleaned = remove_code_blocks(response)
            parsed = json.loads(cleaned, strict=False)
            categorized = parsed.get("categorized_memories", [])
        except Exception as e:
            logger.error(f"Error in LLM categorization: {e}")
            categorized = []

        # Build a lookup of valid memory IDs from this batch
        valid_ids = {item["id"] for item in items}

        results = []
        for entry in categorized:
            memory_id = entry.get("memory_id", "")
            cats = entry.get("categories", [])
            if memory_id not in valid_ids or not isinstance(cats, list):
                continue
            # Normalize and persist
            cats = [c.strip().lower() for c in cats if isinstance(c, str) and c.strip()]
            if cats:
                self.category_db.set_memory_categories(memory_id, cats)
                results.append({"memory_id": memory_id, "categories": cats})

        # For any items that didn't get categorized, assign "other"
        categorized_ids = {r["memory_id"] for r in results}
        for item in items:
            if item["id"] not in categorized_ids:
                self.category_db.set_memory_categories(item["id"], ["other"])
                results.append({"memory_id": item["id"], "categories": ["other"]})

        return results

    # ----------------------------------------------------------------
    # Management
    # ----------------------------------------------------------------

    def reset(self) -> None:
        self.category_db.reset()

    def close(self) -> None:
        self.category_db.close()
