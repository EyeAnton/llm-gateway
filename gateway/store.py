"""Request journal: table `requests` in Postgres (main path) or SQLite (tests / local).

DATABASE_URL examples:
  postgresql+asyncpg://gateway:gateway@db:5432/gateway
  sqlite+aiosqlite:///./gateway.db
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Integer, String, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RequestLog(Base):
    __tablename__ = "requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    key_name: Mapped[str] = mapped_column(String(64), index=True)  # key's name, never the secret
    team: Mapped[str] = mapped_column(String(64), index=True)
    model: Mapped[str] = mapped_column(String(128))  # what the client asked for
    route: Mapped[str | None] = mapped_column(String(64))
    rule: Mapped[str | None] = mapped_column(String(64))
    provider: Mapped[str | None] = mapped_column(String(64))  # who actually answered
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[int] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(String(256))
    latency_ms: Mapped[float] = mapped_column(Float)


class Store:
    def __init__(self, url: str):
        self.engine = create_async_engine(url)
        self.session: async_sessionmaker[AsyncSession] = async_sessionmaker(self.engine, expire_on_commit=False)

    async def init(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def reset(self) -> None:
        """Drop and recreate the journal (tests only)."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    async def log(self, **fields: Any) -> None:
        fields.setdefault("ts", datetime.now(timezone.utc))
        async with self.session() as s:
            s.add(RequestLog(**fields))
            await s.commit()

    async def spent_usd(self, key_name: str) -> float:
        async with self.session() as s:
            q = select(func.coalesce(func.sum(RequestLog.cost_usd), 0.0)).where(RequestLog.key_name == key_name)
            return float((await s.execute(q)).scalar_one())

    async def rows(self, limit: int = 100) -> list[RequestLog]:
        async with self.session() as s:
            q = select(RequestLog).order_by(RequestLog.id.desc()).limit(limit)
            return list((await s.execute(q)).scalars())

    async def stats(self) -> dict[str, Any]:
        def agg(col):
            return select(
                col,
                func.count(),
                func.coalesce(func.sum(RequestLog.cost_usd), 0.0),
                func.coalesce(func.sum(RequestLog.prompt_tokens + RequestLog.completion_tokens), 0),
                func.sum(func.cast(RequestLog.cache_hit, Integer)),
                func.avg(RequestLog.latency_ms),
            ).group_by(col)

        out: dict[str, Any] = {}
        async with self.session() as s:
            for label, col in (("by_key", RequestLog.key_name), ("by_team", RequestLog.team),
                               ("by_provider", RequestLog.provider)):
                out[label] = {
                    str(name): {
                        "requests": n,
                        "cost_usd": round(float(cost), 8),
                        "tokens": int(tokens),
                        "cache_hits": int(hits or 0),
                        "avg_latency_ms": round(float(lat or 0), 1),
                    }
                    for name, n, cost, tokens, hits, lat in (await s.execute(agg(col))).all()
                }
        return out
