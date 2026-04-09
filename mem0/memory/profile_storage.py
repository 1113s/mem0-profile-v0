import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_ATTRIBUTES_PER_SCHEMA = 50


class ProfileSQLiteManager:
    """SQLite storage for profile schemas and user profile data."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS profile_schemas (
                        id              TEXT PRIMARY KEY,
                        name            TEXT NOT NULL UNIQUE,
                        description     TEXT,
                        attributes_json TEXT NOT NULL,
                        created_at      DATETIME,
                        updated_at      DATETIME
                    )
                    """
                )
                self.connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS profile_data (
                        id              TEXT PRIMARY KEY,
                        schema_id       TEXT NOT NULL,
                        user_id         TEXT NOT NULL,
                        attribute_name  TEXT NOT NULL,
                        attribute_value TEXT,
                        created_at      DATETIME,
                        updated_at      DATETIME,
                        UNIQUE(schema_id, user_id, attribute_name)
                    )
                    """
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to create profile tables: {e}")
                raise

    # ----------------------------------------------------------------
    # Profile Schema CRUD
    # ----------------------------------------------------------------

    def create_schema(
        self,
        name: str,
        description: Optional[str],
        attributes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if len(attributes) > MAX_ATTRIBUTES_PER_SCHEMA:
            raise ValueError(
                f"A profile schema can have at most {MAX_ATTRIBUTES_PER_SCHEMA} attributes, got {len(attributes)}."
            )

        # Validate attribute name uniqueness
        attr_names = [a["name"] for a in attributes]
        if len(attr_names) != len(set(attr_names)):
            raise ValueError("Attribute names must be unique within a schema.")

        schema_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        attributes_json = json.dumps(attributes, ensure_ascii=False)

        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    INSERT INTO profile_schemas (id, name, description, attributes_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (schema_id, name, description, attributes_json, now, now),
                )
                self.connection.execute("COMMIT")
            except sqlite3.IntegrityError:
                self.connection.execute("ROLLBACK")
                raise ValueError(f"A profile schema with name '{name}' already exists.")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to create profile schema: {e}")
                raise

        return self._build_schema_dict(schema_id, name, description, attributes, now, now)

    def get_schema(self, schema_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self.connection.execute(
                "SELECT id, name, description, attributes_json, created_at, updated_at FROM profile_schemas WHERE id = ?",
                (schema_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_schema_dict(row)

    def get_schema_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self.connection.execute(
                "SELECT id, name, description, attributes_json, created_at, updated_at FROM profile_schemas WHERE name = ?",
                (name,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_schema_dict(row)

    def list_schemas(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.connection.execute(
                "SELECT id, name, description, attributes_json, created_at, updated_at FROM profile_schemas ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = cur.fetchall()
        return [self._row_to_schema_dict(row) for row in rows]

    def update_schema(
        self,
        schema_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        add_attributes: Optional[List[Dict[str, Any]]] = None,
        update_attributes: Optional[List[Dict[str, Any]]] = None,
        delete_attribute_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        schema = self.get_schema(schema_id)
        if schema is None:
            raise ValueError(f"Profile schema '{schema_id}' not found.")

        current_attributes = schema["attributes"]
        attr_map = {a["name"]: a for a in current_attributes}

        # Delete attributes
        if delete_attribute_names:
            for attr_name in delete_attribute_names:
                attr_map.pop(attr_name, None)
            # Also delete associated profile data
            self._delete_profile_data_for_attributes(schema_id, delete_attribute_names)

        # Update existing attributes
        if update_attributes:
            for attr in update_attributes:
                if attr["name"] in attr_map:
                    attr_map[attr["name"]] = attr

        # Add new attributes
        if add_attributes:
            for attr in add_attributes:
                if attr["name"] not in attr_map:
                    attr_map[attr["name"]] = attr

        new_attributes = list(attr_map.values())
        if len(new_attributes) > MAX_ATTRIBUTES_PER_SCHEMA:
            raise ValueError(
                f"A profile schema can have at most {MAX_ATTRIBUTES_PER_SCHEMA} attributes, would have {len(new_attributes)}."
            )

        now = datetime.now(timezone.utc).isoformat()
        new_name = name if name is not None else schema["name"]
        new_description = description if description is not None else schema["description"]
        attributes_json = json.dumps(new_attributes, ensure_ascii=False)

        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    UPDATE profile_schemas
                    SET name = ?, description = ?, attributes_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (new_name, new_description, attributes_json, now, schema_id),
                )
                self.connection.execute("COMMIT")
            except sqlite3.IntegrityError:
                self.connection.execute("ROLLBACK")
                raise ValueError(f"A profile schema with name '{new_name}' already exists.")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to update profile schema: {e}")
                raise

        return self._build_schema_dict(schema_id, new_name, new_description, new_attributes, schema["created_at"], now)

    def delete_schema(self, schema_id: str) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute("DELETE FROM profile_data WHERE schema_id = ?", (schema_id,))
                self.connection.execute("DELETE FROM profile_schemas WHERE id = ?", (schema_id,))
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to delete profile schema: {e}")
                raise

    # ----------------------------------------------------------------
    # User Profile Data
    # ----------------------------------------------------------------

    def get_user_profile(self, schema_id: str, user_id: str) -> Dict[str, Any]:
        """Get a user's profile, merging schema defaults with extracted values."""
        schema = self.get_schema(schema_id)
        if schema is None:
            raise ValueError(f"Profile schema '{schema_id}' not found.")

        # Fetch extracted values
        with self._lock:
            cur = self.connection.execute(
                "SELECT attribute_name, attribute_value FROM profile_data WHERE schema_id = ? AND user_id = ?",
                (schema_id, user_id),
            )
            rows = cur.fetchall()
        extracted = {row[0]: row[1] for row in rows}

        # Merge: schema attributes + extracted values
        attributes = []
        for attr in schema["attributes"]:
            value = extracted.get(attr["name"], attr.get("default_value"))
            attributes.append({
                "name": attr["name"],
                "description": attr["description"],
                "value": value,
            })

        return {
            "schema_id": schema_id,
            "schema_name": schema["name"],
            "user_id": user_id,
            "attributes": attributes,
        }

    def upsert_profile_attribute(
        self,
        schema_id: str,
        user_id: str,
        attribute_name: str,
        attribute_value: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                # Check if the record already exists
                cur = self.connection.execute(
                    "SELECT id FROM profile_data WHERE schema_id = ? AND user_id = ? AND attribute_name = ?",
                    (schema_id, user_id, attribute_name),
                )
                existing = cur.fetchone()
                if existing:
                    self.connection.execute(
                        "UPDATE profile_data SET attribute_value = ?, updated_at = ? WHERE id = ?",
                        (attribute_value, now, existing[0]),
                    )
                else:
                    self.connection.execute(
                        """
                        INSERT INTO profile_data (id, schema_id, user_id, attribute_name, attribute_value, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (str(uuid.uuid4()), schema_id, user_id, attribute_name, attribute_value, now, now),
                    )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to upsert profile attribute: {e}")
                raise

    def delete_user_profile(self, schema_id: str, user_id: str) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    "DELETE FROM profile_data WHERE schema_id = ? AND user_id = ?",
                    (schema_id, user_id),
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to delete user profile: {e}")
                raise

    def delete_all_user_profiles(self, user_id: str) -> None:
        """Delete all profile data for a user across all schemas."""
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    "DELETE FROM profile_data WHERE user_id = ?",
                    (user_id,),
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to delete all user profiles: {e}")
                raise

    # ----------------------------------------------------------------
    # Management
    # ----------------------------------------------------------------

    def reset(self) -> None:
        """Drop and recreate all profile tables."""
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute("DROP TABLE IF EXISTS profile_data")
                self.connection.execute("DROP TABLE IF EXISTS profile_schemas")
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to reset profile tables: {e}")
                raise
        self._create_tables()

    def close(self) -> None:
        if self.connection:
            self.connection.close()
            self.connection = None

    def __del__(self):
        self.close()

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _delete_profile_data_for_attributes(self, schema_id: str, attribute_names: List[str]) -> None:
        if not attribute_names:
            return
        placeholders = ",".join("?" for _ in attribute_names)
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    f"DELETE FROM profile_data WHERE schema_id = ? AND attribute_name IN ({placeholders})",
                    [schema_id] + attribute_names,
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to delete profile data for attributes: {e}")
                raise

    @staticmethod
    def _row_to_schema_dict(row) -> Dict[str, Any]:
        return {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "attributes": json.loads(row[3]),
            "created_at": row[4],
            "updated_at": row[5],
        }

    @staticmethod
    def _build_schema_dict(
        schema_id: str,
        name: str,
        description: Optional[str],
        attributes: List[Dict[str, Any]],
        created_at: str,
        updated_at: str,
    ) -> Dict[str, Any]:
        return {
            "id": schema_id,
            "name": name,
            "description": description,
            "attributes": attributes,
            "created_at": created_at,
            "updated_at": updated_at,
        }
