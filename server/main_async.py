import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from mem0 import AsyncMemory
from mem0.memory.profile_manager import ProfileManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")

MIN_KEY_LENGTH = 16

if not ADMIN_API_KEY:
    logging.warning(
        "ADMIN_API_KEY not set - API endpoints are UNSECURED! "
        "Set ADMIN_API_KEY environment variable for production use."
    )
else:
    if len(ADMIN_API_KEY) < MIN_KEY_LENGTH:
        logging.warning(
            "ADMIN_API_KEY is shorter than %d characters - consider using a longer key for production.",
            MIN_KEY_LENGTH,
        )
    logging.info("API key authentication enabled")


# Vector store config (supports pgvector and qdrant)
VECTOR_STORE_PROVIDER = os.environ.get("VECTOR_STORE_PROVIDER", "qdrant")

# pgvector config
# POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
# POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
# POSTGRES_DB = os.environ.get("POSTGRES_DB", "postgres")
# POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
# POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
# POSTGRES_COLLECTION_NAME = os.environ.get("POSTGRES_COLLECTION_NAME", "memories")

# Qdrant config
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "mem0")

# Graph store config
ENABLE_GRAPH = os.environ.get("ENABLE_GRAPH", "false").lower() in ("true", "1", "yes")
# NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
# NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
# NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mem0graph")

# MEMGRAPH_URI = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
# MEMGRAPH_USERNAME = os.environ.get("MEMGRAPH_USERNAME", "memgraph")
# MEMGRAPH_PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "mem0graph")

# LLM config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-nano-2025-04-14")

EMBEDDER_URL = os.environ.get("EMBEDDER_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBEDDER_PROVIDER = os.environ.get("EMBEDDER_PROVIDER")
EMBEDDER_API_KEY = os.environ.get("EMBEDDER_API_KEY")
EMBEDDER_MODEL = os.environ.get("EMBEDDER_MODEL", "text-embedding-3-small")
EMBEDDER_DIMS = os.environ.get("EMBEDDER_DIMS", 1536)


HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "/app/history/history.db")

# Build vector store config based on provider
if VECTOR_STORE_PROVIDER == "qdrant":
    _vector_store_config = {
        "provider": "qdrant",
        "config": {
            "host": QDRANT_HOST,
            "port": QDRANT_PORT,
            "collection_name": QDRANT_COLLECTION_NAME,
        },
    }
# else:
#    _vector_store_config = {
#        "provider": "pgvector",
#        "config": {
#            "host": POSTGRES_HOST,
#            "port": int(POSTGRES_PORT),
#            "dbname": POSTGRES_DB,
#            "user": POSTGRES_USER,
#            "password": POSTGRES_PASSWORD,
#            "collection_name": POSTGRES_COLLECTION_NAME,
#        },
#    }

DEFAULT_CONFIG = {
    "version": "v1.1",
    "vector_store": _vector_store_config,
    "llm": {
        "provider": LLM_PROVIDER,
        "config": {
            "api_key": OPENAI_API_KEY,
            "temperature": 0.2,
            "model": LLM_MODEL,
            "openai_base_url": OPENAI_BASE_URL,
        },
    },
    "embedder": {
        "provider": EMBEDDER_PROVIDER or LLM_PROVIDER,
        "config": {
            "api_key": EMBEDDER_API_KEY,
            "model": EMBEDDER_MODEL,
            "openai_base_url": EMBEDDER_URL,
            "embedding_dims": EMBEDDER_DIMS,
        },
    },
    "history_db_path": HISTORY_DB_PATH,
}

if ENABLE_GRAPH:
    DEFAULT_CONFIG["graph_store"] = {
        "provider": "neo4j",
        "config": {"url": NEO4J_URI, "username": NEO4J_USERNAME, "password": NEO4J_PASSWORD},
    }


# Global instances, initialized in lifespan
MEMORY_INSTANCE: Optional[AsyncMemory] = None
PROFILE_MANAGER: Optional[ProfileManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize AsyncMemory and ProfileManager on startup, cleanup on shutdown."""
    global MEMORY_INSTANCE, PROFILE_MANAGER
    MEMORY_INSTANCE = await AsyncMemory.from_config(DEFAULT_CONFIG)
    PROFILE_MANAGER = ProfileManager(
        db_path=HISTORY_DB_PATH,
        llm_config=DEFAULT_CONFIG["llm"],
    )
    logging.info("AsyncMemory and ProfileManager initialized successfully.")
    yield
    MEMORY_INSTANCE = None
    if PROFILE_MANAGER:
        PROFILE_MANAGER.close()
        PROFILE_MANAGER = None
    logging.info("AsyncMemory and ProfileManager released.")


app = FastAPI(
    title="Mem0 REST APIs (Async)",
    description=(
        "A REST API for managing and searching memories for your AI Agents and Apps.\n\n"
        "## Authentication\n"
        "When the ADMIN_API_KEY environment variable is set, all endpoints require "
        "the `X-API-Key` header for authentication."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    """Validate the API key when ADMIN_API_KEY is configured. No-op otherwise."""
    if ADMIN_API_KEY:
        if api_key is None:
            raise HTTPException(
                status_code=401,
                detail="X-API-Key header is required.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        if not secrets.compare_digest(api_key, ADMIN_API_KEY):
            raise HTTPException(
                status_code=401,
                detail="Invalid API key.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
    return api_key


class Message(BaseModel):
    role: str = Field(..., description="Role of the message (user or assistant).")
    content: str = Field(..., description="Message content.")


class MemoryCreate(BaseModel):
    messages: List[Message] = Field(..., description="List of messages to store.")
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    infer: Optional[bool] = Field(None, description="Whether to extract facts from messages. Defaults to True.")
    memory_type: Optional[str] = Field(None, description="Type of memory to store (e.g. 'core').")
    prompt: Optional[str] = Field(None, description="Custom prompt to use for fact extraction.")
    profile_schema_id: Optional[str] = Field(None, description="Profile schema ID for structured attribute extraction.")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query.")
    user_id: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = Field(None, description="Maximum number of results to return.")
    threshold: Optional[float] = Field(None, description="Minimum similarity score for results.")


# --- Profile Schema Models ---

class ProfileAttributeCreate(BaseModel):
    name: str = Field(..., description="Attribute name (e.g. 'age', 'occupation').")
    description: str = Field(..., description="Attribute description to guide extraction.")
    default_value: Optional[str] = Field(None, description="Initial default value.")


class ProfileSchemaCreate(BaseModel):
    name: str = Field(..., description="Unique schema name.")
    description: Optional[str] = Field(None, description="Schema description.")
    attributes: List[ProfileAttributeCreate] = Field(..., description="List of attributes to extract.")


class ProfileSchemaUpdate(BaseModel):
    name: Optional[str] = Field(None, description="New schema name.")
    description: Optional[str] = Field(None, description="New schema description.")
    add_attributes: Optional[List[ProfileAttributeCreate]] = Field(None, description="Attributes to add.")
    update_attributes: Optional[List[ProfileAttributeCreate]] = Field(None, description="Attributes to update.")
    delete_attribute_names: Optional[List[str]] = Field(None, description="Attribute names to delete.")


@app.post("/configure", summary="Configure Mem0")
async def set_config(config: Dict[str, Any], _api_key: Optional[str] = Depends(verify_api_key)):
    """Set memory configuration."""
    global MEMORY_INSTANCE
    MEMORY_INSTANCE = await AsyncMemory.from_config(config)
    return {"message": "Configuration set successfully"}


@app.post("/memories", summary="Create memories")
async def add_memory(memory_create: MemoryCreate, _api_key: Optional[str] = Depends(verify_api_key)):
    """Store new memories. Optionally extract user profile in parallel."""
    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier (user_id, agent_id, run_id) is required.")

    # Build memory params (exclude profile_schema_id which is handled separately)
    params = {
        k: v for k, v in memory_create.model_dump().items()
        if v is not None and k not in ("messages", "profile_schema_id")
    }
    try:
        messages = [m.model_dump() for m in memory_create.messages]
        memory_coro = MEMORY_INSTANCE.add(messages=messages, **params)

        # Run profile extraction in parallel if requested
        profile_task = None
        if memory_create.profile_schema_id and memory_create.user_id:
            profile_task = asyncio.to_thread(
                PROFILE_MANAGER.extract,
                messages,
                memory_create.user_id,
                memory_create.profile_schema_id,
            )

        if profile_task:
            mem_result, prof_result = await asyncio.gather(memory_coro, profile_task)
            mem_result["profile"] = prof_result
        else:
            mem_result = await memory_coro

        return JSONResponse(content=mem_result)
    except Exception as e:
        logging.exception("Error in add_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories", summary="Get memories")
async def get_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Retrieve stored memories."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        return await MEMORY_INSTANCE.get_all(**params)
    except Exception as e:
        logging.exception("Error in get_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}", summary="Get a memory")
async def get_memory(memory_id: str, _api_key: Optional[str] = Depends(verify_api_key)):
    """Retrieve a specific memory by ID."""
    try:
        return await MEMORY_INSTANCE.get(memory_id)
    except Exception as e:
        logging.exception("Error in get_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", summary="Search memories")
async def search_memories(search_req: SearchRequest, _api_key: Optional[str] = Depends(verify_api_key)):
    """Search for memories based on a query."""
    try:
        params = {k: v for k, v in search_req.model_dump().items() if v is not None and k != "query"}
        return await MEMORY_INSTANCE.search(query=search_req.query, **params)
    except Exception as e:
        logging.exception("Error in search_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/memories/{memory_id}", summary="Update a memory")
async def update_memory(memory_id: str, updated_memory: Dict[str, Any], _api_key: Optional[str] = Depends(verify_api_key)):
    """Update an existing memory with new content.

    Args:
        memory_id (str): ID of the memory to update
        updated_memory (str): New content to update the memory with

    Returns:
        dict: Success message indicating the memory was updated
    """
    try:
        return await MEMORY_INSTANCE.update(memory_id=memory_id, data=updated_memory)
    except Exception as e:
        logging.exception("Error in update_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}/history", summary="Get memory history")
async def memory_history(memory_id: str, _api_key: Optional[str] = Depends(verify_api_key)):
    """Retrieve memory history."""
    try:
        return await MEMORY_INSTANCE.history(memory_id=memory_id)
    except Exception as e:
        logging.exception("Error in memory_history:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories/{memory_id}", summary="Delete a memory")
async def delete_memory(memory_id: str, _api_key: Optional[str] = Depends(verify_api_key)):
    """Delete a specific memory by ID."""
    try:
        await MEMORY_INSTANCE.delete(memory_id=memory_id)
        return {"message": "Memory deleted successfully"}
    except Exception as e:
        logging.exception("Error in delete_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories", summary="Delete all memories")
async def delete_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Delete all memories for a given identifier."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        await MEMORY_INSTANCE.delete_all(**params)
        return {"message": "All relevant memories deleted"}
    except Exception as e:
        logging.exception("Error in delete_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", summary="Reset all memories")
async def reset_memory(_api_key: Optional[str] = Depends(verify_api_key)):
    """Completely reset stored memories."""
    try:
        await MEMORY_INSTANCE.reset()
        return {"message": "All memories reset"}
    except Exception as e:
        logging.exception("Error in reset_memory:")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------
# Profile Schema Endpoints
# ----------------------------------------------------------------


@app.post("/profile_schemas", summary="Create a profile schema")
async def create_profile_schema(
    schema_create: ProfileSchemaCreate,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Create a new profile schema (template) for structured attribute extraction."""
    try:
        result = await asyncio.to_thread(
            PROFILE_MANAGER.create_schema,
            name=schema_create.name,
            description=schema_create.description,
            attributes=[attr.model_dump() for attr in schema_create.attributes],
        )
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.exception("Error in create_profile_schema:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/profile_schemas", summary="List profile schemas")
async def list_profile_schemas(
    limit: int = 100,
    offset: int = 0,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """List all profile schemas."""
    try:
        result = await asyncio.to_thread(PROFILE_MANAGER.list_schemas, limit=limit, offset=offset)
        return JSONResponse(content=result)
    except Exception as e:
        logging.exception("Error in list_profile_schemas:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/profile_schemas/{schema_id}", summary="Get a profile schema")
async def get_profile_schema(
    schema_id: str,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Get a specific profile schema by ID."""
    try:
        result = await asyncio.to_thread(PROFILE_MANAGER.get_schema, schema_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Profile schema not found.")
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error in get_profile_schema:")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/profile_schemas/{schema_id}", summary="Update a profile schema")
async def update_profile_schema(
    schema_id: str,
    schema_update: ProfileSchemaUpdate,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Update a profile schema (add/update/delete attributes, rename, etc.)."""
    try:
        result = await asyncio.to_thread(
            PROFILE_MANAGER.update_schema,
            schema_id,
            name=schema_update.name,
            description=schema_update.description,
            add_attributes=[a.model_dump() for a in schema_update.add_attributes] if schema_update.add_attributes else None,
            update_attributes=[a.model_dump() for a in schema_update.update_attributes] if schema_update.update_attributes else None,
            delete_attribute_names=schema_update.delete_attribute_names,
        )
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.exception("Error in update_profile_schema:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/profile_schemas/{schema_id}", summary="Delete a profile schema")
async def delete_profile_schema(
    schema_id: str,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Delete a profile schema and all associated profile data."""
    try:
        await asyncio.to_thread(PROFILE_MANAGER.delete_schema, schema_id)
        return {"message": "Profile schema deleted successfully"}
    except Exception as e:
        logging.exception("Error in delete_profile_schema:")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------
# User Profile Endpoints
# ----------------------------------------------------------------


@app.get("/profiles/{schema_id}/users/{user_id}", summary="Get user profile")
async def get_user_profile(
    schema_id: str,
    user_id: str,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Get a user's extracted profile for a given schema."""
    try:
        result = await asyncio.to_thread(PROFILE_MANAGER.get_user_profile, schema_id, user_id)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.exception("Error in get_user_profile:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/profiles/{schema_id}/users/{user_id}", summary="Delete user profile")
async def delete_user_profile(
    schema_id: str,
    user_id: str,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Delete a user's profile data for a given schema."""
    try:
        await asyncio.to_thread(PROFILE_MANAGER.delete_user_profile, schema_id, user_id)
        return {"message": "User profile deleted successfully"}
    except Exception as e:
        logging.exception("Error in delete_user_profile:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", summary="Redirect to the OpenAPI documentation", include_in_schema=False)
def home():
    """Redirect to the OpenAPI documentation."""
    return RedirectResponse(url="/docs")
