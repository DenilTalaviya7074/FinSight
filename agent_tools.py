from langchain.tools import tool
from db_models import SessionLocal, Company, KPI, Ratio, Risk
from sqlalchemy import and_
from rag_utils import retrieve_chunks

@tool
def get_kpi(company_name: str, kpi_name: str) -> float:
    """Retrieve a KPI value for a company."""
    db = SessionLocal()
    company = db.query(Company).filter(Company.name == company_name).first()
    if not company:
        db.close()
        return None
    kpi = db.query(KPI).filter(and_(KPI.company_id == company.id, KPI.kpi_name == kpi_name)).first()
    db.close()
    return kpi.value if kpi else None

