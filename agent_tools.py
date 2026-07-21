from langchain.tools import tool
from db_models import SessionLocal, Company, KPI, Ratio, Risk
from sqlalchemy import and_
from rag_utils import retrieve_chunks

# NOTE: create_react_agent (plain ReAct) only supports a SINGLE STRING as
# "Action Input" — it cannot map multiple named parameters. Rather than
# fighting LangChain's structured-chat / ChatOllama compatibility issues,
# these tools now accept one string and parse it themselves. Expected
# format: "company_name, field_name" (comma-separated). This is much
# easier for a small local model (e.g. llama3.2:3b) to produce reliably
# than nested JSON.

def _parse_two_args(tool_input: str):
    """Split 'Apple, total_revenue' -> ('Apple', 'total_revenue')."""
    if "," not in tool_input:
        raise ValueError(
            f"Expected input as 'company_name, field_name' but got: {tool_input!r}"
        )
    company_name, field_name = tool_input.split(",", 1)
    return company_name.strip().strip('"').strip("'"), field_name.strip().strip('"').strip("'")


@tool
def get_kpi(tool_input: str) -> float:
    """Retrieve a KPI value for a company.
    Input MUST be a single string formatted as: "company_name, kpi_name"
    Example: "Apple, total_revenue" """
    company_name, kpi_name = _parse_two_args(tool_input)
    db = SessionLocal()
    company = db.query(Company).filter(Company.name == company_name).first()
    if not company:
        db.close()
        return None
    kpi = db.query(KPI).filter(and_(KPI.company_id == company.id, KPI.kpi_name == kpi_name)).first()
    db.close()
    return kpi.value if kpi else None


@tool
def calculate_ratio(tool_input: str) -> float:
    """Calculate a financial ratio for a company.
    Input MUST be a single string formatted as: "company_name, ratio_name"
    Valid ratio_name values: 'current_ratio', 'debt_to_equity', 'roe', 'gross_margin', 'net_margin'
    Example: "Apple, current_ratio" """
    company_name, ratio_name = _parse_two_args(tool_input)
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
    """Get all risk factors for a company. Input is just the company name."""
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
def retrieve_context(tool_input: str) -> str:
    """Retrieve relevant text chunks from a company's annual report.
    Input MUST be a single string formatted as: "company_name, query"
    Example: "Apple, supply chain risk" """
    company_name, query = _parse_two_args(tool_input)
    chunks = retrieve_chunks(query, company_name, k=3)
    return "\n\n".join(chunks) if chunks else "No relevant information found."

# (Optional) A comparison tool can be added later.