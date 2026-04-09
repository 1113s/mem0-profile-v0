import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CATEGORIES = [
    ("personal", "家庭、朋友、个人生活"),
    ("preferences", "喜好、偏好、习惯"),
    ("work", "职业、公司、工作内容"),
    ("health", "健康、健身、饮食、睡眠"),
    ("education", "学历、课程、技能学习"),
    ("finance", "收入、支出、投资、理财"),
    ("relationships", "社交关系、家人、朋友"),
    ("travel", "旅行、出行、地点偏好"),
    ("hobbies", "兴趣爱好、休闲活动"),
    ("food", "饮食偏好、餐厅、烹饪"),
    ("technology", "技术偏好、工具、设备"),
    ("goals", "个人和职业目标"),
    ("events", "日程、活动、约会"),
    ("shopping", "购物偏好、消费记录"),
    ("entertainment", "影视、音乐、书籍、游戏"),
    ("communication", "沟通风格、联系方式"),
    ("habits", "日常习惯、作息规律"),
    ("opinions", "观点、态度、信念"),
    ("skills", "技能、能力、专长"),
    ("other", "无法归入其他类别的信息"),
]


class CategorySQLiteManager:
    """SQLite storage for memory categories and memory-category associations."""

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
                    CREATE TABLE IF NOT EXISTS categories (
                        id          TEXT PRIMARY KEY,
                        name        TEXT NOT NULL UNIQUE,
                        description TEXT,
                        created_at  DATETIME,
                        updated_at  DATETIME
                    )
                    """
                )
                self.connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_categories (
                        id          TEXT PRIMARY KEY,
                        memory_id   TEXT NOT NULL,
                        category_id TEXT NOT NULL,
                        created_at  DATETIME,
                        UNIQUE(memory_id, category_id)
                    )
                    """
                )
                self.connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_mc_memory_id ON memory_categories(memory_id)"
                )
                self.connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_mc_category_id ON memory_categories(category_id)"
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to create category tables: {e}")
                raise

    # ----------------------------------------------------------------
    # Seed
    # ----------------------------------------------------------------

    def seed_default_categories(self) -> None:
        """Insert default categories idempotently."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                for name, description in DEFAULT_CATEGORIES:
                    self.connection.execute(
                        "INSERT OR IGNORE INTO categories (id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                        (str(uuid.uuid4()), name, description, now, now),
                    )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to seed default categories: {e}")
                raise

    # ----------------------------------------------------------------
    # Category CRUD
    # ----------------------------------------------------------------

    def create_category(self, name: str, description: Optional[str] = None) -> Dict[str, Any]:
        category_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    "INSERT INTO categories (id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (category_id, name.strip().lower(), description, now, now),
                )
                self.connection.execute("COMMIT")
            except sqlite3.IntegrityError:
                self.connection.execute("ROLLBACK")
                raise ValueError(f"Category '{name}' already exists.")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to create category: {e}")
                raise
        return {"id": category_id, "name": name.strip().lower(), "description": description, "created_at": now, "updated_at": now}

    def get_category_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self.connection.execute(
                "SELECT id, name, description, created_at, updated_at FROM categories WHERE name = ?",
                (name.strip().lower(),),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {"id": row[0], "name": row[1], "description": row[2], "created_at": row[3], "updated_at": row[4]}

    def list_categories(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.connection.execute(
                "SELECT id, name, description, created_at, updated_at FROM categories ORDER BY name LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = cur.fetchall()
        return [{"id": r[0], "name": r[1], "description": r[2], "created_at": r[3], "updated_at": r[4]} for r in rows]

    def delete_category(self, category_id: str) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute("DELETE FROM memory_categories WHERE category_id = ?", (category_id,))
                self.connection.execute("DELETE FROM categories WHERE id = ?", (category_id,))
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to delete category: {e}")
                raise

    def update_category(self, category_id: str, name: Optional[str] = None, description: Optional[str] = None) -> Dict[str, Any]:
        """Update a category's name and/or description."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self.connection.execute(
                "SELECT id, name, description, created_at, updated_at FROM categories WHERE id = ?",
                (category_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"Category '{category_id}' not found.")

            new_name = name.strip().lower() if name is not None else row[1]
            new_description = description if description is not None else row[2]

            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    "UPDATE categories SET name = ?, description = ?, updated_at = ? WHERE id = ?",
                    (new_name, new_description, now, category_id),
                )
                self.connection.execute("COMMIT")
            except sqlite3.IntegrityError:
                self.connection.execute("ROLLBACK")
                raise ValueError(f"Category '{new_name}' already exists.")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to update category: {e}")
                raise
        return {"id": category_id, "name": new_name, "description": new_description, "created_at": row[3], "updated_at": now}

    # ----------------------------------------------------------------
    # Memory-Category Associations
    # ----------------------------------------------------------------

    def set_memory_categories(self, memory_id: str, category_names: List[str]) -> None:
        """Set categories for a memory. Replaces existing associations. Auto-creates unknown categories."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                # Remove existing associations
                self.connection.execute("DELETE FROM memory_categories WHERE memory_id = ?", (memory_id,))
                # Resolve each category name and insert association
                for name in category_names:
                    name = name.strip().lower()
                    if not name:
                        continue
                    cur = self.connection.execute("SELECT id FROM categories WHERE name = ?", (name,))
                    row = cur.fetchone()
                    if row:
                        category_id = row[0]
                    else:
                        # Auto-create new category
                        category_id = str(uuid.uuid4())
                        self.connection.execute(
                            "INSERT INTO categories (id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                            (category_id, name, f"Automatically created category for {name}", now, now),
                        )
                    self.connection.execute(
                        "INSERT OR IGNORE INTO memory_categories (id, memory_id, category_id, created_at) VALUES (?, ?, ?, ?)",
                        (str(uuid.uuid4()), memory_id, category_id, now),
                    )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to set memory categories: {e}")
                raise

    def remove_memory_categories(self, memory_id: str) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute("DELETE FROM memory_categories WHERE memory_id = ?", (memory_id,))
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to remove memory categories: {e}")
                raise

    def remove_memory_categories_batch(self, memory_ids: List[str]) -> None:
        """Remove category associations for multiple memories at once."""
        if not memory_ids:
            return
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                placeholders = ",".join("?" for _ in memory_ids)
                self.connection.execute(
                    f"DELETE FROM memory_categories WHERE memory_id IN ({placeholders})",
                    memory_ids,
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to batch remove memory categories: {e}")
                raise

    def get_categories_for_memory(self, memory_id: str) -> List[str]:
        with self._lock:
            cur = self.connection.execute(
                """
                SELECT c.name FROM categories c
                JOIN memory_categories mc ON mc.category_id = c.id
                WHERE mc.memory_id = ?
                ORDER BY c.name
                """,
                (memory_id,),
            )
            rows = cur.fetchall()
        return [r[0] for r in rows]

    def get_memories_by_category(self, category_name: str, limit: int = 100, offset: int = 0) -> List[str]:
        with self._lock:
            cur = self.connection.execute(
                """
                SELECT mc.memory_id FROM memory_categories mc
                JOIN categories c ON c.id = mc.category_id
                WHERE c.name = ?
                ORDER BY mc.created_at DESC
                LIMIT ? OFFSET ?
                """,
                (category_name.strip().lower(), limit, offset),
            )
            rows = cur.fetchall()
        return [r[0] for r in rows]

    def get_memory_ids_by_categories(self, category_names: List[str]) -> List[str]:
        """Get all memory IDs belonging to any of the given categories (OR semantics)."""
        if not category_names:
            return []
        names = [n.strip().lower() for n in category_names]
        placeholders = ",".join("?" for _ in names)
        with self._lock:
            cur = self.connection.execute(
                f"""
                SELECT DISTINCT mc.memory_id FROM memory_categories mc
                JOIN categories c ON c.id = mc.category_id
                WHERE c.name IN ({placeholders})
                """,
                names,
            )
            rows = cur.fetchall()
        return [r[0] for r in rows]

    def get_category_counts(self) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.connection.execute(
                """
                SELECT c.name, COUNT(mc.memory_id) as count
                FROM categories c
                LEFT JOIN memory_categories mc ON mc.category_id = c.id
                GROUP BY c.id, c.name
                HAVING count > 0
                ORDER BY count DESC
                """
            )
            rows = cur.fetchall()
        return [{"name": r[0], "count": r[1]} for r in rows]

    # ----------------------------------------------------------------
    # Management
    # ----------------------------------------------------------------

    def reset(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute("DROP TABLE IF EXISTS memory_categories")
                self.connection.execute("DROP TABLE IF EXISTS categories")
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to reset category tables: {e}")
                raise
        self._create_tables()

    def close(self) -> None:
        if self.connection:
            self.connection.close()
            self.connection = None

    def __del__(self):
        self.close()
