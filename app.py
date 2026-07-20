import streamlit as st
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from agent_tools import get_kpi, calculate_ratio, get_risks, retrieve_context
from db_models import SessionLocal, Company, KPI, Ratio, Risk
from rag_utils import ingest_document
from extractors import extract_kpis, extract_risks
import os
import uuid
import tempfile
import plotly.express as px
import pandas as pd

# Set page config
st.set_page_config(page_title="FinSight AI", layout="wide")
st.title("📊 FinSight AI — Conversational Financial Analyst")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "companies" not in st.session_state:
    st.session_state.companies = []
if "agent_executor" not in st.session_state:
    # We'll create the agent once and reuse it
    llm = ChatOllama(model="llama3.2:3b", temperature=0)
    tools = [get_kpi, calculate_ratio, get_risks, retrieve_context]
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a financial analyst assistant. Use the provided tools to answer questions about companies.
        If you don't know, say so. Be concise."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = create_react_agent(llm, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )

# Sidebar: Upload and company list
with st.sidebar:
    st.header("Upload Annual Report")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    company_name = st.text_input("Company Name")
    ticker = st.text_input("Ticker (optional)")
    report_year = st.number_input("Report Year", min_value=2000, max_value=2030, step=1, value=2024)

    if st.button("Upload & Process"):
        if uploaded_file and company_name:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner("Processing report... (may take a minute)"):
                # 1. Extract text for KPI/risk extraction
                from rag_utils import extract_text_from_pdf
                raw_text = extract_text_from_pdf(tmp_path)

                # 2. Extract KPIs and risks using LLM
                kpis = extract_kpis(raw_text, company_name)
                risks = extract_risks(raw_text, company_name)

                # 3. Ingest into ChromaDB
                ingest_document(tmp_path, company_name, ticker, report_year, kpis, risks)

                # 4. Store in SQLite
                db = SessionLocal()
                company_id = str(uuid.uuid4())
                company = Company(id=company_id, name=company_name, ticker=ticker, report_year=report_year)
                db.add(company)
                for k in kpis:
                    db.add(KPI(company_id=company_id, kpi_name=k["name"], value=k["value"], unit=k.get("unit"), confidence=k.get("confidence", 1.0)))
                for r in risks:
                    db.add(Risk(company_id=company_id, category=r["category"], severity=r["severity"], description=r["description"]))
                db.commit()
                db.close()

                # Cleanup temp file
                os.unlink(tmp_path)

            st.success(f"✅ {company_name} uploaded and processed!")
        else:
            st.error("Please provide company name and file.")

    # List companies in DB
    st.subheader("Companies in Database")
    db = SessionLocal()
    companies = db.query(Company).all()
    db.close()
    if companies:
        for c in companies:
            st.write(f"- {c.name} ({c.ticker or 'N/A'}) {c.report_year}")
    else:
        st.write("No companies yet.")

# Main chat area
st.subheader("💬 Chat with FinSight")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Ask a financial question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke the agent
                result = st.session_state.agent_executor.invoke({"input": prompt})
                reply = result.get("output", "No response.")
            except Exception as e:
                reply = f"Error: {e}"
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

# Dashboard: Show KPIs for a selected company (optional)
st.sidebar.subheader("📊 Quick Dashboard")
if companies:
    selected_company = st.sidebar.selectbox("Select company", [c.name for c in companies])
    if selected_company:
        db = SessionLocal()
        company = db.query(Company).filter(Company.name == selected_company).first()
        if company:
            kpis = db.query(KPI).filter(KPI.company_id == company.id).all()
            ratios = db.query(Ratio).filter(Ratio.company_id == company.id).all()
            db.close()
            if kpis:
                df = pd.DataFrame([{"KPI": k.kpi_name, "Value": k.value, "Unit": k.unit} for k in kpis])
                st.sidebar.dataframe(df)
                # simple bar chart for numeric KPIs
                fig = px.bar(df, x="KPI", y="Value", title=f"{selected_company} KPIs")
                st.sidebar.plotly_chart(fig, use_container_width=True)
            else:
                st.sidebar.write("No KPIs extracted yet.")