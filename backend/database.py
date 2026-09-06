import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/signatures.db")

# Ensure data directory exists for SQLite
if "sqlite" in DATABASE_URL:
    os.makedirs("./data", exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class VerificationLog(Base):
    __tablename__ = "verification_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    verdict = Column(String(50), nullable=False)
    overall_score = Column(Float, nullable=False)
    metric_confidence = Column(Float, nullable=False)
    keypoint_confidence = Column(Float, nullable=False)
    cnn_similarity = Column(Float, default=0.0)
    ref_corners = Column(Integer, default=0)
    test_corners = Column(Integer, default=0)
    ref_crest_trough = Column(Float, default=0.0)
    test_crest_trough = Column(Float, default=0.0)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()