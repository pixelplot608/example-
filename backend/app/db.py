import os
from typing import Optional

import asyncpg


DATABASE_URL = os.environ.get("DATABASE_URL")

_pool: Optional[asyncpg.pool.Pool] = None


async def get_pool() -> asyncpg.pool.Pool:
    """
    Lazily create and return a global connection pool.

    You must set the DATABASE_URL environment variable before starting
    the backend, e.g.:

    postgresql://postgres:YOUR_PASSWORD@db.dwpkeknhsmdltkulrqep.supabase.co:5432/postgres
    """
    global _pool

    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL environment variable is not set. "
                "Set it to your Supabase Postgres connection string."
            )
        _pool = await asyncpg.create_pool(DATABASE_URL)

    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None

