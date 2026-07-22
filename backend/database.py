import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Load the database URL from environment variable
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_ovELxGzVkt29@ep-weathered-flower-aslph1ag.c-4.eu-central-1.aws.neon.tech/rag_chatbot?sslmode=require"
)

# Create the engine
engine = create_engine(DATABASE_URL)

# Session Local factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative Base for models
Base = declarative_base()

# FastAPI Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
