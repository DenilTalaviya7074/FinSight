from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

llm = ChatOllama(model='llama3.2:latest', temperature=0)


# --- Helper: locate the relevant section instead of blindly truncating ---
def _extract_relevant_section(text: str, keywords: list, window: int = 6000) -> str:
    """
    Search for the first occurrence (case-insensitive) of any keyword in the
    document and return a window of text starting a bit before that match.

    This replaces the old text[:4000] truncation, which for a multi-page
    SEC filing (10-Q/10-K) only captured the cover page and table of
    contents -- never the actual financial statements or risk factors.

    If none of the keywords are found, falls back to the first `window`
    characters so extraction still runs on something rather than failing.
    """
    text_lower = text.lower()
    for kw in keywords:
        idx = text_lower.find(kw.lower())
        if idx != -1:
            start = max(0, idx - 200)
            return text[start:start + window]
    return text[:window]


# --- KPI extraction ---
class KPIItem(BaseModel):
    name: str = Field(description="KPI name")
    value: float = Field(description="numeric value")
    unit: Optional[str] = Field(None, description="unit")
    confidence: float = Field(1.0, description="confidence 0-1")

class KPIList(BaseModel):
    kpis: List[KPIItem]

kpi_parser = PydanticOutputParser(pydantic_object=KPIList)

kpi_prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract these KPIs if present: revenue, net_income, operating_income, total_assets, total_liabilities, total_equity, current_assets, current_liabilities, gross_profit, cash_from_operations, eps, employees. Return as JSON list. Include unit if known. Use the most recent/current period figures if multiple periods are shown (prefer the latest quarter or fiscal year column, not prior-year comparatives)."""),
    ("human", "Text:\n{text}")
])

kpi_chain = kpi_prompt | llm | kpi_parser

# Keywords that mark where the actual financial statements live in a
# 10-K/10-Q. Balance sheet and income statement keywords are searched
# separately since they're usually on different pages, and we combine
# both windows so revenue/net income AND assets/liabilities/equity are
# all available to the model in one shot.
_BALANCE_SHEET_KEYWORDS = [
    "CONSOLIDATED BALANCE SHEETS",
    "CONSOLIDATED BALANCE SHEET",
]
_INCOME_STATEMENT_KEYWORDS = [
    "CONSOLIDATED STATEMENTS OF INCOME",
    "CONSOLIDATED STATEMENTS OF OPERATIONS",
    "CONSOLIDATED STATEMENT OF INCOME",
]

def extract_kpis(text: str, company: str) -> List[dict]:
    try:
        balance_sheet_section = _extract_relevant_section(text, _BALANCE_SHEET_KEYWORDS, window=4000)
        income_statement_section = _extract_relevant_section(text, _INCOME_STATEMENT_KEYWORDS, window=4000)

        # Combine both sections so the model sees revenue/net income
        # alongside assets/liabilities/equity in a single extraction call.
        combined_text = (
            "=== INCOME STATEMENT SECTION ===\n"
            f"{income_statement_section}\n\n"
            "=== BALANCE SHEET SECTION ===\n"
            f"{balance_sheet_section}"
        )

        result = kpi_chain.invoke({"text": combined_text})
        return [k.dict() for k in result.kpis]
    except Exception as e:
        print(f"KPI extraction error: {e}")
        return []


# --- Risk extraction ---
class RiskItem(BaseModel):
    category: str = Field(description="e.g., Market, Financial")
    severity: str = Field(description="High/Medium/Low")
    description: str = Field(description="brief description")

class RiskList(BaseModel):
    risks: List[RiskItem]

risk_parser = PydanticOutputParser(pydantic_object=RiskList)

risk_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract business risks from the report. Classify each as High/Medium/Low. Return JSON list."),
    ("human", "Text:\n{text}")
])

risk_chain = risk_prompt | llm | risk_parser

# "Item 1A" / "Risk Factors" is the standard SEC filing section header for
# risk disclosures in both 10-Ks and 10-Qs.
_RISK_KEYWORDS = [
    "ITEM 1A. RISK FACTORS",
    "ITEM 1A RISK FACTORS",
    "Risk Factors",
]

def extract_risks(text: str, company: str) -> List[dict]:
    try:
        section = _extract_relevant_section(text, _RISK_KEYWORDS, window=6000)
        result = risk_chain.invoke({"text": section})
        return [r.dict() for r in result.risks]
    except Exception as e:
        print(f"Risk extraction error: {e}")
        return []