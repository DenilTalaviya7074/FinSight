from sqlalchemy import create_engine, Column, String, Integer, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Use a local SQLite file
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "finsight.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Company(Base):
    __tablename__ = "companies"
    id = Column(String, primary_key=True)
    name = Column(String)
    ticker = Column(String, nullable=True)
    report_year = Column(Integer)

class KPI(Base):
    __tablename__ = "kpis"
    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(String)
    kpi_name = Column(String)
    value = Column(Float)
    unit = Column(String, nullable=True)
    confidence = Column(Float, default=1.0)

class Ratio(Base):
    __tablename__ = "ratios"
    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(String)
    ratio_name = Column(String)
    value = Column(Float)

class Risk(Base):
    __tablename__ = "risks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(String)
    category = Column(String)
    severity = Column(String)
    description = Column(Text)

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)