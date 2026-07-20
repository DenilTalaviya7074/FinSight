import streamlit as st
import os
import tempfile
import pandas as pd
from db_models import SessionLocal, Company, KPI, Risk
from rag_utils import ingest_document, retrieve_chunks, extract_text_from_pdf
from extractors import extract_kpis, extract_risks
from agent_tools import get_kpi, calculate_ratio, get_risks, retrieve_context

# -------------------- Page config --------------------
st.set_page_config(page_title="FinSight - Financial Document QA", layout="wide")
st.title("📊 FinSight: Financial Report Analysis with RAG & LLM")

# -------------------- Initialize DB --------------------
def get_companies():
    db = SessionLocal()
    companies = db.query(Company).all()
    db.close()
    return companies

companies = get_companies()

# -------------------- Sidebar: Ingest / Manage --------------------
with st.sidebar:
    st.header("📁 Document Management")
    upload_option = st.radio("Choose action:", ["Upload new PDF", "Use existing files (if any)"])

    if upload_option == "Upload new PDF":
        uploaded_file = st.file_uploader("Choose a 10-K/10-Q PDF file", type="pdf")
        company_name = st.text_input("Company name (e.g., Apple Inc.)")
        ticker = st.text_input("Ticker symbol (optional)")
        report_year = st.number_input("Report year", min_value=2000, max_value=2030, step=1, value=2024)
        if st.button("Ingest PDF") and uploaded_file and company_name:
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name

            try:
                # Ingest into RAG
                ingest_document(temp_path, company_name, ticker, report_year, kpi_list=[], risk_list=[])

                # Extract KPIs and risks
                text = extract_text_from_pdf(temp_path)
                if not text.strip():
                    st.warning("Could not extract text from PDF. Please check the file.")
                else:
                    kpis = extract_kpis(text, company_name)
                    risks = extract_risks(text, company_name)

                    # Store in DB
                    db = SessionLocal()
                    # Add company if not exists
                    existing = db.query(Company).filter(Company.name == company_name).first()
                    if not existing:
                        # Use a simple hash as id (or you can use uuid)
                        import hashlib
                        comp_id = hashlib.md5(company_name.encode()).hexdigest()
                        comp = Company(id=comp_id, name=company_name, ticker=ticker, report_year=report_year)
                        db.add(comp)
                        db.commit()
                        db.refresh(comp)
                        company_id = comp.id
                    else:
                        company_id = existing.id

                    # Store KPIs
                    for k in kpis:
                        kpi_obj = KPI(
                            company_id=company_id,
                            kpi_name=k['name'],
                            value=k['value'],
                            unit=k.get('unit'),
                            confidence=k.get('confidence', 1.0)
                        )
                        db.add(kpi_obj)

                    # Store risks
                    for r in risks:
                        risk_obj = Risk(
                            company_id=company_id,
                            category=r['category'],
                            severity=r['severity'],
                            description=r['description']
                        )
                        db.add(risk_obj)

                    db.commit()
                    db.close()
                    st.success(f"✅ Successfully ingested {company_name}!")
            except Exception as e:
                st.error(f"Error during ingestion: {e}")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    else:
        st.info("If you have already ingested files, select a company below to query.")

    st.divider()
    st.header("🔍 Query & Tools")
    companies = get_companies()
    company_names = [c.name for c in companies] if companies else []
    if company_names:
        selected_company = st.selectbox("Select company", company_names)
    else:
        selected_company = None
        st.warning("No companies in database. Please ingest a PDF first.")

# -------------------- Main area: Query --------------------
if selected_company:
    st.subheader(f"Querying: {selected_company}")

    # Show basic info
    db = SessionLocal()
    comp = db.query(Company).filter(Company.name == selected_company).first()
    if comp:
        st.write(f"**Ticker:** {comp.ticker}  |  **Report Year:** {comp.report_year}")
    db.close()

    # Tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["🔎 Ask a Question", "📈 KPIs", "📉 Risks", "🧮 Ratios"])

    with tab1:
        query = st.text_area("Enter your question about the company:", height=100)
        if st.button("Get Answer"):
            if query:
                with st.spinner("Retrieving context..."):
                    context = retrieve_context(selected_company, query)
                if context and "No relevant information" not in context:
                    st.subheader("📄 Retrieved Context:")
                    st.write(context)
                else:
                    st.info("No relevant information found in the report.")
            else:
                st.warning("Please enter a question.")

    with tab2:
        st.subheader("Extracted Key Performance Indicators (KPIs)")
        db = SessionLocal()
        comp = db.query(Company).filter(Company.name == selected_company).first()
        if comp:
            kpis = db.query(KPI).filter(KPI.company_id == comp.id).all()
            if kpis:
                data = [{"KPI": k.kpi_name, "Value": k.value, "Unit": k.unit or "", "Confidence": k.confidence} for k in kpis]
                df = pd.DataFrame(data)
                st.dataframe(df)
            else:
                st.info("No KPIs extracted yet.")
        db.close()

    with tab3:
        st.subheader("Risks")
        risk_text = get_risks(selected_company)
        st.text(risk_text)

    with tab4:
        st.subheader("Financial Ratios")
        ratio_options = ["current_ratio", "debt_to_equity", "roe", "gross_margin", "net_margin"]
        ratio_choice = st.selectbox("Select ratio", ratio_options)
        if st.button("Calculate"):
            result = calculate_ratio(selected_company, ratio_choice)
            if result is not None:
                st.success(f"{ratio_choice.replace('_', ' ').title()} = {result:.2f}")
            else:
                st.error("Could not calculate ratio. Missing required KPI values.")