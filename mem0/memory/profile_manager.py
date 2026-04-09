import json
import logging
from typing import Any, Dict, List, Optional

from mem0.configs.prompts import PROFILE_EXTRACTION_PROMPT
from mem0.memory.profile_storage import ProfileSQLiteManager
from mem0.memory.utils import parse_messages, remove_code_blocks
from mem0.utils.factory import LlmFactory

logger = logging.getLogger(__name__)


class ProfileManager:
    """Independent user profile manager, decoupled from Memory/AsyncMemory.

    Manages profile schemas (templates) and extracts structured user attributes
    from conversations via LLM.
    """

    def __init__(self, db_path: str, llm_config: dict):
        """
        Args:
            db_path: SQLite database path (can share the same file as history.db).
            llm_config: LLM configuration dict, e.g. {"provider": "openai", "config": {...}}.
        """
        self.profile_db = ProfileSQLiteManager(db_path)
        self.llm = LlmFactory.create(llm_config["provider"], llm_config.get("config", {}))

    # ----------------------------------------------------------------
    # Profile Schema CRUD (delegates to ProfileSQLiteManager)
    # ----------------------------------------------------------------

    def create_schema(self, name: str, description: Optional[str] = None, attributes: Optional[List[dict]] = None) -> dict:
        """Create a new profile schema."""
        return self.profile_db.create_schema(name, description, attributes or [])

    def get_schema(self, schema_id: str) -> Optional[dict]:
        """Get a profile schema by ID."""
        return self.profile_db.get_schema(schema_id)

    def list_schemas(self, limit: int = 100, offset: int = 0) -> list:
        """List all profile schemas."""
        return self.profile_db.list_schemas(limit=limit, offset=offset)

    def update_schema(
        self,
        schema_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        add_attributes: Optional[List[dict]] = None,
        update_attributes: Optional[List[dict]] = None,
        delete_attribute_names: Optional[List[str]] = None,
    ) -> dict:
        """Update a profile schema (add/update/delete attributes, rename, etc.)."""
        return self.profile_db.update_schema(
            schema_id,
            name=name,
            description=description,
            add_attributes=add_attributes,
            update_attributes=update_attributes,
            delete_attribute_names=delete_attribute_names,
        )

    def delete_schema(self, schema_id: str) -> None:
        """Delete a profile schema and all associated profile data."""
        self.profile_db.delete_schema(schema_id)

    # ----------------------------------------------------------------
    # User Profile
    # ----------------------------------------------------------------

    def get_user_profile(self, schema_id: str, user_id: str) -> dict:
        """Get a user's profile for a given schema."""
        return self.profile_db.get_user_profile(schema_id, user_id)

    def delete_user_profile(self, schema_id: str, user_id: str) -> None:
        """Delete a user's profile data for a given schema."""
        self.profile_db.delete_user_profile(schema_id, user_id)

    def delete_all_user_profiles(self, user_id: str) -> None:
        """Delete all profile data for a user across all schemas."""
        self.profile_db.delete_all_user_profiles(user_id)

    # ----------------------------------------------------------------
    # Core: Profile Extraction
    # ----------------------------------------------------------------

    def extract(self, messages: List[Dict[str, Any]], user_id: str, profile_schema_id: str) -> dict:
        """Extract structured user profile attributes from conversation messages.

        Args:
            messages: List of message dicts (role + content).
            user_id: The user whose profile is being extracted.
            profile_schema_id: ID of the profile schema to use.

        Returns:
            dict with schema_id, user_id, and updated_attributes.
        """
        schema = self.profile_db.get_schema(profile_schema_id)
        if schema is None:
            raise ValueError(f"Profile schema '{profile_schema_id}' not found.")

        existing_profile = self.profile_db.get_user_profile(profile_schema_id, user_id)
        existing_values = {
            attr["name"]: attr["value"]
            for attr in existing_profile["attributes"]
            if attr["value"] is not None
        }

        schema_attrs_desc = [
            {"name": attr["name"], "description": attr["description"]}
            for attr in schema["attributes"]
        ]

        parsed = parse_messages(messages)
        prompt_text = PROFILE_EXTRACTION_PROMPT.format(
            schema_attributes=json.dumps(schema_attrs_desc, ensure_ascii=False),
            existing_profile=json.dumps(existing_values, ensure_ascii=False),
        )

        response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": f"Conversation:\n{parsed}"},
            ],
            response_format={"type": "json_object"},
        )

        try:
            cleaned = remove_code_blocks(response)
            extracted = json.loads(cleaned, strict=False).get("profile", {})
        except Exception as e:
            logger.error(f"Error parsing profile extraction response: {e}")
            extracted = {}

        valid_attr_names = {attr["name"] for attr in schema["attributes"]}
        updated_attributes = {}
        for attr_name, attr_value in extracted.items():
            if attr_name in valid_attr_names and attr_value is not None:
                self.profile_db.upsert_profile_attribute(
                    profile_schema_id, user_id, attr_name, str(attr_value)
                )
                updated_attributes[attr_name] = str(attr_value)

        return {
            "schema_id": profile_schema_id,
            "user_id": user_id,
            "updated_attributes": updated_attributes,
        }

    # ----------------------------------------------------------------
    # Management
    # ----------------------------------------------------------------

    def reset(self) -> None:
        """Reset all profile schemas and data."""
        self.profile_db.reset()

    def close(self) -> None:
        """Close the database connection."""
        self.profile_db.close()
