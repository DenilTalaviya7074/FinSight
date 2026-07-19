from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

llm = ChatOllama(model='llama3.2:latest', temperature=0)

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
    ("system", """Extract these KPIs if present: revenue, net_income, operating_income, total_assets, total_liabilities, total_equity, cash_from_operations, eps, employees. Return as JSON list. Include unit if known."""),
    ("human", "Text:\n{text}")
])

kpi_chain = kpi_prompt | llm | kpi_parser

def extract_kpis(text: str, company: str) -> List[dict]:
    try:
        # Limit text length to avoid context overflow
        result = kpi_chain.invoke({"text": text[:4000]})
        return [k.dict() for k in result.kpis]
    except Exception as e:
        print(f"KPI extraction error: {e}")
        return []