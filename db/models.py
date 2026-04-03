from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import Boolean
from sqlalchemy import DateTime
from sqlalchemy import Enum as SAEnum
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String
from sqlalchemy import UniqueConstraint
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship


class Base(DeclarativeBase):
    pass


class TrainingSessionStatus(str, Enum):
    RUNNING = "running"
    FAILED = "failed"
    COMPLETED = "completed"


class TrainingSession(Base):
    __tablename__ = "training_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    steps: Mapped[int | None] = mapped_column(nullable=True)
    episodes: Mapped[int | None] = mapped_column(nullable=True)
    best_reward: Mapped[float | None] = mapped_column(nullable=True)

    status: Mapped[TrainingSessionStatus] = mapped_column(
        SAEnum(TrainingSessionStatus, name="training_session_status"),
        nullable=False,
        default=TrainingSessionStatus.RUNNING,
    )
    finished: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    tracks: Mapped[list[str]] = mapped_column(
        ARRAY(String),
        nullable=False,
        default=list,
    )

    policies: Mapped[list["Policy"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    eval_runs: Mapped[list["EvalRun"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )


class Policy(Base):
    __tablename__ = "policies"
    __table_args__ = (
        UniqueConstraint("session_id", "version", name="uq_policy_session_version"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("training_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version: Mapped[int] = mapped_column(nullable=False)
    path: Mapped[str] = mapped_column(String(1024), nullable=False)

    session: Mapped["TrainingSession"] = relationship(back_populates="policies")
    eval_runs: Mapped[list["EvalRun"]] = relationship(
        back_populates="policy",
        cascade="all, delete-orphan",
    )


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    session_id: Mapped[int] = mapped_column(
        ForeignKey("training_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    policy_id: Mapped[int | None] = mapped_column(
        ForeignKey("policies.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    track_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    best_lap: Mapped[float | None] = mapped_column(nullable=True)
    worst_lap: Mapped[float | None] = mapped_column(nullable=True)
    avg_lap: Mapped[float | None] = mapped_column(nullable=True)

    collisions: Mapped[int] = mapped_column(nullable=False, default=0)
    off_track: Mapped[int] = mapped_column(nullable=False, default=0)

    collision_locations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    session: Mapped["TrainingSession"] = relationship(back_populates="eval_runs")
    policy: Mapped["Policy | None"] = relationship(back_populates="eval_runs")
