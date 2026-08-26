import json
import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, String, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DB_URL = os.getenv("DATABASE_URL", "sqlite:///razorsentry.db")

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


# ORM model representing a single immutable fraud decision row in the audit table
class DecisionRecord(Base):
    __tablename__ = "decisions"

    decision_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    transaction_id = Column(String, nullable=False, index=True)
    score = Column(Float, nullable=False)
    decision = Column(String, nullable=False)
    top_reasons = Column(String, nullable=False)
    latency_ms = Column(Float, nullable=False)
    model_version = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    transaction_type = Column(String, nullable=False)


# Converts a DecisionRecord ORM object to a plain Python dict
def _record_to_dict(record: DecisionRecord) -> dict:
    return {
        "decision_id": record.decision_id,
        "timestamp": record.timestamp.isoformat() if record.timestamp else None,
        "transaction_id": record.transaction_id,
        "score": record.score,
        "decision": record.decision,
        "top_reasons": json.loads(record.top_reasons),
        "latency_ms": record.latency_ms,
        "model_version": record.model_version,
        "amount": record.amount,
        "transaction_type": record.transaction_type,
    }


# Creates the decisions table in the database if it does not already exist
def init_db() -> None:
    Base.metadata.create_all(bind=engine)


# Inserts one decision row into the audit table and returns the generated decision_id
def log_decision(
    transaction_id: str,
    score: float,
    decision: str,
    top_reasons: list[str],
    latency_ms: float,
    model_version: str,
    amount: float,
    transaction_type: str,
) -> str:
    decision_id = str(uuid.uuid4())
    record = DecisionRecord(
        decision_id=decision_id,
        timestamp=datetime.now(timezone.utc),
        transaction_id=transaction_id,
        score=score,
        decision=decision,
        top_reasons=json.dumps(top_reasons),
        latency_ms=latency_ms,
        model_version=model_version,
        amount=amount,
        transaction_type=transaction_type,
    )
    db: Session = SessionLocal()
    try:
        db.add(record)
        db.commit()
    finally:
        db.close()
    return decision_id


# Fetches a single decision row by its UUID and returns it as a dict, or None if not found
def get_decision(decision_id: str) -> dict | None:
    db: Session = SessionLocal()
    try:
        record = db.query(DecisionRecord).filter(
            DecisionRecord.decision_id == decision_id
        ).first()
        return _record_to_dict(record) if record else None
    finally:
        db.close()


# Returns the most recent N decisions ordered by timestamp descending as a list of dicts
def get_recent_decisions(limit: int = 50) -> list[dict]:
    db: Session = SessionLocal()
    try:
        records = (
            db.query(DecisionRecord)
            .order_by(DecisionRecord.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [_record_to_dict(r) for r in records]
    finally:
        db.close()
