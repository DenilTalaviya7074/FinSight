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

@tool
def calculate_ratio(company_name: str, ratio_name: str) -> float:
    """Calculate a financial ratio: 'current_ratio', 'debt_to_equity', 'roe', 'gross_margin', 'net_margin'."""
    db = SessionLocal()
    company = db.query(Company).filter(Company.name == company_name).first()
    if not company:
        db.close()
        return None

    def get_val(name):
        k = db.query(KPI).filter(and_(KPI.company_id == company.id, KPI.kpi_name == name)).first()
        return k.value if k else None

    ca = get_val("current_assets") or get_val("total_current_assets")
    cl = get_val("current_liabilities") or get_val("total_current_liabilities")
    ta = get_val("total_assets")
    tl = get_val("total_liabilities")
    te = get_val("total_equity")
    rev = get_val("revenue")
    ni = get_val("net_income")
    gp = get_val("gross_profit")
    oi = get_val("operating_income")

    result = None
    if ratio_name == "current_ratio" and ca and cl:
        result = ca / cl
    elif ratio_name == "debt_to_equity" and tl and te:
        result = tl / te
    elif ratio_name == "roe" and ni and te:
        result = ni / te
    elif ratio_name == "gross_margin" and gp and rev:
        result = gp / rev
    elif ratio_name == "net_margin" and ni and rev:
        result = ni / rev
    db.close()
    return result

@tool
def get_risks(company_name: str) -> str:
    """Get all risk factors for a company."""
    db = SessionLocal()
    company = db.query(Company).filter(Company.name == company_name).first()
    if not company:
        db.close()
        return "Company not found"
    risks = db.query(Risk).filter(Risk.company_id == company.id).all()
    db.close()
    if not risks:
        return "No risks found"
    return "\n".join([f"- [{r.severity}] {r.category}: {r.description}" for r in risks])

@tool
def retrieve_context(company_name: str, query: str) -> str:
    """Retrieve relevant text chunks from the company's annual report."""
    chunks = retrieve_chunks(query, company_name, k=3)
    return "\n\n".join(chunks) if chunks else "No relevant information found."

# (Optional) A comparison tool can be added later.