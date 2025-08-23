#####################################################################################################
# Python Module for Gen AI Solutions - Month End Process                                            #
# Author: Subhadip Kundu (Jade Global)                                                              #
# --------------------------------------------------------------------------------------------------#
#    Date      |     Author          |                   Comment                                    #
# ------------ + ------------------- + ------------------------------------------------------------ #
# 15-Jun-2024  | Subhadip Kundu      | Created the Initial Code                                     #
# 20-Jun-2024  | Subhadip Kundu      | Added Workflow and Dashboard Details                         #
# 24-Jun-2024  | Subhadip Kundu      | Change the Dashboard Details                                 #
# 20-Aug-2024  | Subhadip Kundu      | Change the Vector Database from PineCone to FAISS            #
#####################################################################################################

import os
import time
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
import snowflake.connector
from langchain.chains import RetrievalQA
from PIL import Image
from io import StringIO
from tabulate import tabulate
import plotly.express as px
from streamlit_option_menu import option_menu

from requests.auth import HTTPBasicAuth
import requests
from FewShotSettings import few_shot_settings
from ZeroShotAnalyzeSettings import zero_shot_analyze_settings
import UiPath_API_Queue_Load
import prompts
import countdown

# Setup the Page
jadeimage = Image.open("assets/jadeglobalsmall.png")
st.set_page_config(page_title="Jade FinClose AI", page_icon=jadeimage, layout="wide")

# Declare Global Variables
load_dotenv()
api_key = st.secrets["OpenAI_Secret_Key"]
st_UserName = st.secrets["streamlit_username"]
st_Password = st.secrets["streamlit_password"]
api_username = st.secrets["api_username"]
api_password = st.secrets["api_password"]
llm_model_name = ""
year_selected = ""
month_selected = ""
prev_month = ""

# Class Few Shot Prompt for Text to SQL
class few_shot_prompt_utility:

    def __init__(self, examples, prefix, suffix, input_variables, example_template, example_variables):
        self.examples = examples
        self.prefix = prefix
        self.suffix = suffix
        self.input_variables = input_variables
        self.example_template = example_template
        self.example_variables = example_variables

    def get_prompt_template(self):
        example_prompt = PromptTemplate(
            input_variables=self.example_variables,
            template=self.example_template
        )
        return example_prompt

    def get_embeddings(self):
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        return embeddings

    def get_example_selector(self, embeddings):
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            self.examples,
            embeddings,
            FAISS,
            k=3
        )
        return example_selector

    def get_prompt(self, question, example_selector, example_prompt):
        prompt_template = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix=self.prefix,
            suffix=self.suffix,
            input_variables=self.input_variables
        )
        return prompt_template


# Class Zero Shot Prompt for Analysis Part
class zero_shot_analyze_utility:

    def __init__(self, question, ask, context, metadata):
        self.question = question
        self.ask = ask
        self.context = context
        self.metadata = metadata

    def get_analyze_prompt(self):
        template, variables = zero_shot_analyze_settings.get_prompt_template(self.ask, self.metadata)
        prompt_template = PromptTemplate(template=template, input_variables=variables)
        prompt_template.format(question=self.question, context=self.context)
        return prompt_template


# Initializing the Few_Shot Utility
def few_shot():
    prefix = few_shot_settings.get_prefix()
    suffix, input_variable = few_shot_settings.get_suffix()
    examples = few_shot_settings.get_examples()
    example_template, example_variables = few_shot_settings.get_example_template()
    fewShot = few_shot_prompt_utility(examples=examples, prefix=prefix, suffix=suffix, input_variables=input_variable,
                                      example_template=example_template, example_variables=example_variables)
    return fewShot


# Initializing the Large Language Model
def large_language_model(model_name):
    llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=2000, openai_api_key=api_key)
    return llm


# Function for Text to Sql
def text_to_sql(user_question):
    try:
        question = user_question
        llm = large_language_model(llm_model_name)
        fewShot = few_shot()
        example_prompt = fewShot.get_prompt_template()
        embeddings = fewShot.get_embeddings()
        example_selector = fewShot.get_example_selector(embeddings)
        prompt_template = fewShot.get_prompt(question, example_selector, example_prompt)
        docsearch = FAISS.load_local("db_faiss_index", embeddings, allow_dangerous_deserialization=True)
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=docsearch.as_retriever(),
                                               chain_type_kwargs={"prompt": prompt_template})
        sql_query = qa_chain({"query": question})['result']
        return sql_query

    except Exception as er:
        return "Error in Text_to_Sql - " + str(er)


# Function to run the SQL query into Snowflake
def run_sql_query(query):
    try:
        snow_con = snowflake.connector.connect(
            account=os.getenv("Snowflake_Account_Name"),
            user=os.getenv("Snowflake_User_Name"),
            password=os.getenv("Snowflake_User_Credential"),
            role=os.getenv("Snowflake_User_Role"),
            warehouse=os.getenv("Snowflake_Warehouse_Name"),
            database=os.getenv("Snowflake_Database_Name"),
            schema=os.getenv("Snowflake_Schema_Name")
        )
        data = pd.read_sql(query, snow_con)
        return data
    except Exception as e:
        out_err = ["Error, Data for the provided question is not available in Database :" + str(e)]
        return out_err


# Function to run the SQL query into Snowflake
def update_sql_query(overall_status, kpi, kpi_status):
    try:
        snow_con = snowflake.connector.connect(
            account=os.getenv("Snowflake_Account_Name"),
            user=os.getenv("Snowflake_User_Name"),
            password=os.getenv("Snowflake_User_Credential"),
            role=os.getenv("Snowflake_User_Role"),
            warehouse=os.getenv("Snowflake_Warehouse_Name"),
            database=os.getenv("Snowflake_Database_Name"),
            schema=os.getenv("Snowflake_Schema_Name")
        )

        snow_cur = snow_con.cursor()

        update_query = f"""
            UPDATE DEMO_DB.BUS_INSIGHTS.OVERALL_STATUS
            SET OVERALL_STATUS = {overall_status},
            {kpi} = {kpi_status}
            WHERE FINANCIAL_YEAR = YEAR(CURRENT_DATE)
            AND UPPER(FINANCIAL_MONTH) = UPPER(MONTHNAME(CURRENT_DATE));
    """
        snow_cur.execute(update_query)
        snow_con.commit()
    except Exception as e:
        out_err = ["Error, Failed to update Status :" + str(e)]
        return out_err


def result_analysis(dataframe, question):
    analysis_question_part1 = '''Provide analysis of the data in tabular format below. \n '''
    analysis_question_prompt = '''\nUse "Ask" and "Metadata" information as supporting data for the analysis. This information is mentioned toward end of this text.
        Keep analysis strictly for business users working in the finance domain to understand nature of output. Limit your response accordingly.
        Few Rules to follow are:
        1. If the result for the query is in tabular format make sure the whole analysis is in same format.
        2. The analysis must be within 80-120 words. 
        3. Do not include supplied data into analysis.
        4. You can add some points from your end if you think that is relevant to the user question. 
        '''
    analysis_question = str(analysis_question_part1) + str(dataframe) + str(analysis_question_prompt)
    fewShot = few_shot()
    llm = large_language_model(llm_model_name)
    embeddings = fewShot.get_embeddings()
    docsearch = FAISS.load_local("db_faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = docsearch.similarity_search(question)
    metadata = ""
    for i in docs:
        metadata = metadata + "\n" + i.page_content
    zeroShotAnlyze = zero_shot_analyze_utility(analysis_question, question, "company finance data", metadata)
    analyze_prompt = zeroShotAnlyze.get_analyze_prompt()
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=docsearch.as_retriever(),
                                           chain_type_kwargs={"prompt": analyze_prompt})
    result = qa_chain({"query": analysis_question})['result']
    return result


def plot_chart(dataFrame):
    df = pd.DataFrame(dataFrame)
    fig = px.bar(df, x=df.columns[0], y=df.columns[-1], color=df.columns[1])
    return st.plotly_chart(fig, width=0, height=300, use_container_width=True)


def format_amount(amount):
    if amount >= 1_000_000:
        return f"$ {amount / 1_000_000:.2f}M"
    elif amount >= 1_000:
        return f"$ {amount / 1_000:.2f}K"
    else:
        return f"$ {amount}"


# @st.fragment(run_every="1s")
@st.experimental_fragment(run_every="1s")
def countdown_timer(curr_first_bd, curr_last_bd, prev_first_bd, prev_last_bd):
    if st.session_state.master_button == 1:
        formatted_first_bd, formatted_last_bd, remain_days, remain_hours, remain_minutes, remain_seconds = countdown.last_bus_day_countdown(curr_first_bd, curr_last_bd)
        st.subheader("Overall Status - :green[**GREEN**]", divider='rainbow')
        st.markdown("")
        st.markdown(f"<h5>Start On    - {formatted_first_bd}</h5>", unsafe_allow_html=True)
        st.markdown(f"<h5>Close On  - {formatted_last_bd}</h5>", unsafe_allow_html=True)
        st.markdown(
            f"<h6>Close in {remain_days} Days {remain_hours:02d}:{remain_minutes:02d}:{remain_seconds:02d}</h6>",
            unsafe_allow_html=True)
    else:
        formatted_first_bd, formatted_last_bd, remain_days, remain_hours, remain_minutes, remain_seconds = countdown.last_bus_day_countdown(prev_first_bd, prev_last_bd)
        st.subheader("Overall Status - :green[**GREEN**]", divider='rainbow')
        st.markdown("")
        st.markdown(f"<h5>Start On    - {formatted_first_bd}</h5>", unsafe_allow_html=True)
        st.markdown(f"<h5>Close On  - {formatted_last_bd}</h5>", unsafe_allow_html=True)
        st.markdown(f"<h6>Close in 0 Days 00:00:00</h6>", unsafe_allow_html=True)


def chat_history(CSV_FILE):
    try:
        chat_history_df = pd.read_csv(CSV_FILE)
        return chat_history_df
    except FileNotFoundError:
        chat_history_df = pd.DataFrame(columns=["User_Chat_History"])
        chat_history_df.to_csv(CSV_FILE, index=False)
        return chat_history_df


try:
    his_file_1 = "Chat_History/jade_Chat_History_Snow.csv"
    his_file_2 = "Chat_History/jade_Chat_History_PDF.csv"
    chat_df_1 = chat_history(his_file_1)
    chat_df_2 = chat_history(his_file_2)
    
    st.markdown(
         """
         <style>
            /* Remove blank space at top and bottom */ 
           .block-container {
               padding-top: 1rem;
               padding-bottom: 0rem;
            }

             [data-testid=stSidebar] img {
                padding-top: 0rem;
               padding-bottom: 0rem;
                 display: block;
                 margin-left: auto;
                 margin-right: auto;
                 width: 55%;
             }
         </style>
         """,
         unsafe_allow_html=True
     )

    if "messages" not in st.session_state.keys():
        st.session_state.messages = []
    if "messages1" not in st.session_state.keys():
        st.session_state.messages1 = []

    # Workflow Status
    if 'master_button' not in st.session_state:
        st.session_state.master_button = 0
    if 'button1' not in st.session_state:
        st.session_state.button1 = 0
    if 'button2' not in st.session_state:
        st.session_state.button2 = 0
    if 'button3' not in st.session_state:
        st.session_state.button3 = 0
    if 'button4' not in st.session_state:
        st.session_state.button4 = 0
    if 'button5' not in st.session_state:
        st.session_state.button5 = 0
    if 'button6' not in st.session_state:
        st.session_state.button6 = 0
    if 'button7' not in st.session_state:
        st.session_state.button7 = 0
    if 'button8' not in st.session_state:
        st.session_state.button8 = 0
    if 'button9' not in st.session_state:
        st.session_state.button9 = 0
    if 'button10' not in st.session_state:
        st.session_state.button10 = 0
    if 'button11' not in st.session_state:
        st.session_state.button11 = 0
    if 'button12' not in st.session_state:
        st.session_state.button12 = 0
    if 'button13' not in st.session_state:
        st.session_state.button13 = 0
    if 'button14' not in st.session_state:
        st.session_state.button14 = 0
    if 'button15' not in st.session_state:
        st.session_state.button15 = 0
    if 'button16' not in st.session_state:
        st.session_state.button16 = 0

    if 'OverallProcess' not in st.session_state:
        st.session_state.OverallProcess = 0
    if 'AP_Process' not in st.session_state:
        st.session_state.AP_Process = 0
        update_sql_query(overall_status=st.session_state.OverallProcess, kpi="AP_STATUS", kpi_status=st.session_state.AP_Process)
    if 'AR_Process' not in st.session_state:
        st.session_state.AR_Process = 0
        update_sql_query(overall_status=st.session_state.OverallProcess, kpi="AR_STATUS", kpi_status=st.session_state.AR_Process)
    if 'GL_Process' not in st.session_state:
        st.session_state.GL_Process = 0
        update_sql_query(overall_status=st.session_state.OverallProcess, kpi="GL_STATUS", kpi_status=st.session_state.GL_Process)
    if 'INV_Process' not in st.session_state:
        st.session_state.INV_Process = 0
        update_sql_query(overall_status=st.session_state.OverallProcess, kpi="INV_STATUS", kpi_status=st.session_state.INV_Process)
    if 'TAX_Process' not in st.session_state:
        st.session_state.TAX_Process = 0
        update_sql_query(overall_status=st.session_state.OverallProcess, kpi="TAX_STATUS", kpi_status=st.session_state.TAX_Process)
    
    # Setup the Widgets
    with st.sidebar:       
        st.image('assets/jadeglobal.png')
        st.markdown("""<h1 style='text-align: center; color: black;'>Financial Close AI</h1>""", unsafe_allow_html=True)

    ### Add Option menu to select the source
    with st.sidebar:
        select_source = option_menu(
            menu_title=None,
            options=['Dashboard', 'Query Financial Data', 'Trigger Month End', 'Business Insights'],
            icons=['graph-up-arrow', 'database', 'robot', 'briefcase'],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "white", "font-size": "14px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "5px 0",
                    "color": "white",
                    "border-radius": "8px",
                    "border": "2px solid #144774",
                    "background-color": "#175388",
                },
                "nav-link-selected": {
                    "background-color": "#ecb713",
                    "color": "white",
                    "font-weight": "normal",
                    "border": "2px solid #c49e10",
                },
            }
        )

        st.markdown("""<p level="2" style="text-align: right; color: black;">powered by boomi</p>""", unsafe_allow_html=True)

    if select_source == 'Dashboard':
        st.title("Dashboard :")
        # Month Map
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }

        # Selling Data
        selling_sql_qry = """SELECT
                                SELLING_YEAR||' '||SELLING_MONTH AS YEAR_MONTH,
                                SELLING_COST AS REVENUE_AMOUNT
                            FROM DEMO_DB.SC_FINCLOSE.SELLING_COST 
                            ORDER BY SELLING_YEAR DESC, 
                            CASE SELLING_MONTH
                                WHEN 'January' THEN 1
                                WHEN 'February' THEN 2
                                WHEN 'March' THEN 3
                                WHEN 'April' THEN 4
                                WHEN 'May' THEN 5
                                WHEN 'June' THEN 6
                                WHEN 'July' THEN 7
                                WHEN 'August' THEN 8
                                WHEN 'September' THEN 9
                                WHEN 'October' THEN 10
                                WHEN 'November' THEN 11
                                WHEN 'December' THEN 12
                            END DESC"""
        selling_sql_result = run_sql_query(selling_sql_qry)
        selling_df = pd.DataFrame(selling_sql_result)
        selling_df[['Year', 'MonthName']] = selling_df['YEAR_MONTH'].str.split(expand=True)
        selling_df['Year'] = selling_df['Year'].astype(int)
        selling_df.columns = selling_df.columns.str.replace('_', ' ')

        # Buying Data
        buying_sql_qry = """SELECT
                                BUYING_YEAR||' '||BUYING_MONTH AS YEAR_MONTH,
                                BUYING_COST AS EXPENSES_AMOUNT
                            FROM DEMO_DB.SC_FINCLOSE.BUYING_COST 
                            ORDER BY BUYING_YEAR DESC, 
                            CASE BUYING_MONTH
                                WHEN 'January' THEN 1
                                WHEN 'February' THEN 2
                                WHEN 'March' THEN 3
                                WHEN 'April' THEN 4
                                WHEN 'May' THEN 5
                                WHEN 'June' THEN 6
                                WHEN 'July' THEN 7
                                WHEN 'August' THEN 8
                                WHEN 'September' THEN 9
                                WHEN 'October' THEN 10
                                WHEN 'November' THEN 11
                                WHEN 'December' THEN 12
                            END DESC"""
        buying_sql_result = run_sql_query(buying_sql_qry)
        buying_df = pd.DataFrame(buying_sql_result)
        buying_df[['Year', 'MonthName']] = buying_df['YEAR_MONTH'].str.split(expand=True)
        buying_df['Year'] = buying_df['Year'].astype(int)
        buying_df.columns = buying_df.columns.str.replace('_', ' ')

        # Distinct Year and Month List
        year_list = list(buying_df.Year.unique())[::-1]
        month_list = list(buying_df.MonthName.unique())

        # Margin Data
        df_merged = pd.merge(selling_df, buying_df, on=['YEAR MONTH', 'Year', 'MonthName'])
        df_merged['Margin Amount'] = df_merged['REVENUE AMOUNT'] - df_merged['EXPENSES AMOUNT']
        df_merged['Margin Amount'] = round((((df_merged['REVENUE AMOUNT'] - df_merged['EXPENSES AMOUNT']) /
                                                df_merged['REVENUE AMOUNT']) * 100), 2)
        df_merged['Month_Num'] = df_merged['MonthName'].map(month_map)
        df_margin = df_merged[['YEAR MONTH', 'Margin Amount', 'Year', 'Month_Num']]

        # # DSO Data
        # dso_sql_qry = """SELECT 
        #                     R.PERIOD_YEAR, 
        #                     R.PERIOD_MONTH,
        #                     R.PERIOD_YEAR||' '||R.PERIOD_MONTH AS YEAR_MONTH,
        #                     ROUND(((R.AMOUNT * DAY(LAST_DAY(TO_DATE(R.PERIOD_MONTH|| ' ' ||R.PERIOD_YEAR, 'MMMM YYYY'))))/S.AMOUNT), 2) AS DSO_AMOUNT
        #                 FROM 
        #                     DEMO_DB.SC_FINCLOSE.DSO AS R
        #                 INNER JOIN
        #                     DEMO_DB.SC_FINCLOSE.DSO AS S
        #                 ON R.PERIOD_YEAR = S.PERIOD_YEAR
        #                 AND R.PERIOD_MONTH = S.PERIOD_MONTH
        #                 AND R.TYPE = 'RECEIVABLES'
        #                 AND S.TYPE = 'CREDIT_SALES';"""
        # dso_sql_result = run_sql_query(dso_sql_qry)
        # dso_df = pd.DataFrame(dso_sql_result)
        # dso_df.columns = dso_df.columns.str.replace('_', ' ')

        # Open PO
        open_PO_qry = """SELECT 
                            CORR_ID AS PO_NUMBER,
                            TO_TIMESTAMP(PO_CREATION_DATE) AS PO_CREATION_DATE,
                            PO_STATUS,
                            VENDOR_NAME,
                            POLOOKUP AS PO_TYPE_CODE,
                            AUTHORIZATION_STATUS,
                            '$ '||PO_AMOUNT AS TOTAL_AMOUNT
                        FROM DEMO_DB.BUS_INSIGHTS.INT_FINCLOSE_BH
                        WHERE PO_STATUS = 'Open'
                        AND VENDOR_NAME != 'Test Vendor'
                        ORDER BY PO_CREATION_DATE DESC
                        LIMIT 100;"""

        # Invoices WIP
        invoices_WIP_qry = """SELECT
                                INVOICE_ID,
                                INVOICE_NUM AS INVOICE_NUMBER,
                                VENDOR_NAME,
                                TO_DATE(SUBSTR(INV_DUEDATE,1,9), 'MM/DD/YYYY') AS INVOICE_DUE_DATE,
                                INVOICE_AMOUNT::NUMBER(10,2) AS INVOICE_AMOUNT,
                                COALESCE(AMOUNT_PAID,0)::NUMBER(10,2) AS AMOUNT_PAID,
                                INVOICE_STATUS
                            FROM DEMO_DB.BUS_INSIGHTS.INT_FINCLOSE_BH
                            WHERE INVOICE_ID IS NOT NULL AND INVOICE_STATUS IN ('Pending Approval')
                            AND VENDOR_NAME != 'Test Vendor';"""

        # Unbilled Revenue
        unbilled_revenue_qry = """SELECT PERIOD, CUSTOMER, AMOUNT, ACTION, COMMENTS
                            FROM
                            (SELECT
                                RELEVANT_PERIOD AS PERIOD,
                                CUSTOMER,
                                '$ '||AMOUNT AS AMOUNT,
                                ACTION,
                                COMMENTS,
                                ROW_NUMBER() OVER(ORDER BY 1) AS RN
                            FROM DEMO_DB.SC_FINCLOSE.UNBILLED_REVENUE)
                            ORDER BY RN DESC;"""
        
        # Current Business Day Query
        curr_busDay_qry = """WITH date_filtered AS (
                                SELECT *
                                FROM DEMO_DB.BUS_INSIGHTS.OVERALL_STATUS
                                WHERE CURRENT_DATE BETWEEN START_DATE AND CLOSE_DATE
                                ORDER BY START_DATE
                                LIMIT 1
                            ), next_start_date AS (
                                SELECT *
                                FROM DEMO_DB.BUS_INSIGHTS.OVERALL_STATUS
                                WHERE START_DATE > CURRENT_DATE
                                ORDER BY START_DATE
                                LIMIT 1
                            )
                            SELECT 
                                START_DATE AS START_ON,
                                CLOSE_DATE AS CLOSE_ON,
                                OVERALL_STATUS,
                                AP_STATUS,
                                AR_STATUS,
                                GL_STATUS,
                                INV_STATUS,
                                TAX_STATUS
                            FROM date_filtered
                                UNION ALL
                            SELECT 
                                START_DATE AS START_ON,
                                CLOSE_DATE AS CLOSE_ON,
                                OVERALL_STATUS,
                                AP_STATUS,
                                AR_STATUS,
                                GL_STATUS,
                                INV_STATUS,
                                TAX_STATUS
                            FROM next_start_date
                            WHERE NOT EXISTS (SELECT 1 FROM date_filtered);"""
        curr_busDay_result = run_sql_query(curr_busDay_qry)
        curr_Start_On = curr_busDay_result.iloc[0]['START_ON']
        curr_Close_On = curr_busDay_result.iloc[0]['CLOSE_ON']
        curr_Overall_Status = float(curr_busDay_result.iloc[0]['OVERALL_STATUS'])
        curr_AP_Status = int(curr_busDay_result.iloc[0]['AP_STATUS'])
        curr_AR_Status = int(curr_busDay_result.iloc[0]['AR_STATUS'])
        curr_GL_Status = int(curr_busDay_result.iloc[0]['GL_STATUS'])
        curr_INV_Status = int(curr_busDay_result.iloc[0]['INV_STATUS'])
        curr_Tax_Status = int(curr_busDay_result.iloc[0]['TAX_STATUS'])
        
        # Previous Business Day Query
        prev_busDay_qry = """SELECT 
                                START_DATE AS START_ON,
                                CLOSE_DATE AS CLOSE_ON,
                                OVERALL_STATUS,
                                AP_STATUS,
                                AR_STATUS,
                                GL_STATUS,
                                INV_STATUS,
                                TAX_STATUS
                            FROM DEMO_DB.BUS_INSIGHTS.OVERALL_STATUS
                            WHERE CLOSE_DATE < CURRENT_DATE
                            ORDER BY CLOSE_DATE DESC
                            LIMIT 1;"""
        prev_busDay_result = run_sql_query(prev_busDay_qry)
        prev_Start_On = prev_busDay_result.iloc[0]['START_ON']
        prev_Close_On = prev_busDay_result.iloc[0]['CLOSE_ON']
        prev_Overall_Status = float(prev_busDay_result.iloc[0]['OVERALL_STATUS'])
        prev_AP_Status = int(prev_busDay_result.iloc[0]['AP_STATUS'])
        prev_AR_Status = int(prev_busDay_result.iloc[0]['AR_STATUS'])
        prev_GL_Status = int(prev_busDay_result.iloc[0]['GL_STATUS'])
        prev_INV_Status = int(prev_busDay_result.iloc[0]['INV_STATUS'])
        prev_Tax_Status = int(prev_busDay_result.iloc[0]['TAX_STATUS'])

        if st.button("Start Month End Process ▶️", key="Start_Month_End_Process", type="primary"):
            st.session_state.master_button = 1

        # Dashboard Main Panel
        col1 = st.columns(2, gap='medium')
        with col1[0]:
            with st.container(border=True, height=380):
                countdown_timer(curr_Start_On, curr_Close_On, prev_Start_On, prev_Close_On)
        # Second Column
        with col1[1]:
            with st.container(border=True, height=380):
                if st.session_state.master_button == 0:
                    st.subheader(f"Status by Function : {round(prev_Overall_Status, 2)}%", divider='rainbow')
                    st.markdown("")
                    st.progress(prev_AP_Status, text=f"Accounts Payable  -  {prev_AP_Status}%")
                    st.progress(prev_AR_Status, text=f"Accounts Receivable  -  {prev_AR_Status}%")
                    st.progress(prev_GL_Status, text=f"General Ledger  -  {prev_GL_Status}%")
                    st.progress(prev_INV_Status, text=f"Inventory  -  {prev_INV_Status}%")
                    st.progress(prev_Tax_Status, text=f"Tax  -  {prev_Tax_Status}%")
                else:
                    # Status Bar for Each of the Period Close
                    st.subheader(f"Status by Function : {round(st.session_state.OverallProcess, 2)}%", divider='rainbow')
                    st.markdown("")
                    AP = st.progress(0, text="Accounts Payable  -  0%")
                    AR = st.progress(0, text="Accounts Receivable  -  0%")
                    GL = st.progress(0, text="General Ledger  -  0%")
                    INV = st.progress(0, text="Inventory  -  0%")
                    TAX = st.progress(0, text="Tax  -  0%")

                    if st.session_state.button1 == 1 or st.session_state.button3 == 1 or st.session_state.button10 == 1 or st.session_state.button12 == 1 or st.session_state.button16 == 1:
                        update_sql_query(overall_status=st.session_state.OverallProcess, kpi="AP_STATUS", kpi_status=st.session_state.AP_Process)
                        AP.progress(st.session_state.AP_Process,
                                    text=f"Accounts Payable  :  {st.session_state.AP_Process}%")
                    if st.session_state.button2 == 1 or st.session_state.button5 == 1 or st.session_state.button6 == 1 or st.session_state.button8 == 1 or st.session_state.button11 == 1:
                        update_sql_query(overall_status=st.session_state.OverallProcess, kpi="AR_STATUS", kpi_status=st.session_state.AR_Process)
                        AR.progress(st.session_state.AR_Process,
                                    text=f"Accounts Receivable  :  {st.session_state.AR_Process}%")
                    if st.session_state.button4 == 1 or st.session_state.button7 == 1:
                        update_sql_query(overall_status=st.session_state.OverallProcess, kpi="GL_STATUS", kpi_status=st.session_state.GL_Process)
                        GL.progress(st.session_state.GL_Process,
                                    text=f"General Ledger  :  {st.session_state.GL_Process}%")
                    if st.session_state.button9 == 1:
                        update_sql_query(overall_status=st.session_state.OverallProcess, kpi="INV_STATUS", kpi_status=st.session_state.INV_Process)
                        INV.progress(st.session_state.INV_Process,
                                        text=f"Inventory  :  {st.session_state.INV_Process}%")
                    if st.session_state.button13 == 1:
                        update_sql_query(overall_status=st.session_state.OverallProcess, kpi="TAX_STATUS", kpi_status=st.session_state.TAX_Process)
                        TAX.progress(st.session_state.TAX_Process,
                                        text=f"Tax  :  {st.session_state.TAX_Process}%")

        col = st.columns(3, gap='medium')
        # First Column
        with col[0]:
            year_selected = st.selectbox("Select a year :", options=year_list)
        # Second Column
        with col[1]:
            month_selected = st.selectbox("Select a month :", options=month_list)
            if month_selected == 'January':
                prev_month = 'January'
            elif month_selected == 'February':
                prev_month = 'January'
            elif month_selected == 'March':
                prev_month = 'February'
            elif month_selected == 'April':
                prev_month = 'March'
            elif month_selected == 'May':
                prev_month = 'April'
            elif month_selected == 'June':
                prev_month = 'May'
            elif month_selected == 'July':
                prev_month = 'June'
            elif month_selected == 'August':
                prev_month = 'July'
        # Third Column
        with col[2]:
            st.markdown('')
            st.markdown('')
            st.markdown('')
            st.markdown('')
            st.markdown('')

        with col[0]:
            st.subheader("Revenue:", divider='rainbow')
            # Metric Graph
            curr_sell_df = \
                selling_df[(selling_df['Year'] == year_selected) & (selling_df['MonthName'] == month_selected)][
                    'REVENUE AMOUNT'].to_frame().reset_index(drop=True)
            curr_sell_data = curr_sell_df.loc[0, 'REVENUE AMOUNT']
            prev_sell_df = \
                selling_df[(selling_df['Year'] == year_selected) & (selling_df['MonthName'] == prev_month)][
                    'REVENUE AMOUNT'].to_frame().reset_index(drop=True)
            prev_sell_data = prev_sell_df.loc[0, 'REVENUE AMOUNT']
            sell_data_diff = str(round((((curr_sell_data - prev_sell_data) / prev_sell_data) * 100), 2)) + '%'
            curr_sell_amount = format_amount(curr_sell_data)
            st.metric(label=str(year_selected) + " " + str(month_selected),
                        value=curr_sell_amount,
                        delta=sell_data_diff)
            # Bar Graph
            fig = px.bar(selling_df,
                            x=selling_df.columns[0],
                            y=selling_df.columns[1],
                            # color=selling_df.columns[1]
                            )
            st.plotly_chart(fig, width=0, height=300, use_container_width=True)

        with col[1]:
            st.subheader("Expenses:", divider='rainbow')
            # Metric Graph
            curr_buy_df = \
                buying_df[(buying_df['Year'] == year_selected) & (buying_df['MonthName'] == month_selected)][
                    'EXPENSES AMOUNT'].to_frame().reset_index(drop=True)
            curr_buy_data = curr_buy_df.loc[0, 'EXPENSES AMOUNT']
            prev_buy_df = \
                buying_df[(buying_df['Year'] == year_selected) & (buying_df['MonthName'] == prev_month)][
                    'EXPENSES AMOUNT'].to_frame().reset_index(drop=True)
            prev_buy_data = prev_buy_df.loc[0, 'EXPENSES AMOUNT']
            buy_data_diff = str(round((((curr_buy_data - prev_buy_data) / prev_buy_data) * 100), 2)) + '%'
            curr_buy_amount = format_amount(curr_buy_data)
            st.metric(label=str(year_selected) + " " + str(month_selected),
                        value=curr_buy_amount,
                        delta=buy_data_diff)
            # Bar Graph
            fig = px.bar(buying_df,
                            x=buying_df.columns[0],
                            y=buying_df.columns[1],
                            # color=buying_df.columns[1]
                            )
            st.plotly_chart(fig, width=0, height=300, use_container_width=True)

        with col[2]:
            # Margin Data
            st.subheader("Margin:", divider='rainbow')
            prev_margin_data = round((((prev_sell_data - prev_buy_data) / prev_buy_data) * 100), 2)
            curr_margin_data = round((((curr_sell_data - curr_buy_data) / curr_buy_data) * 100), 2)
            margin_data_diff = str(
                round((((curr_margin_data - prev_margin_data) / prev_margin_data) * 100), 2)) + '%'
            st.metric(label=str(year_selected) + " " + str(month_selected), value=str(curr_margin_data) + '%',
                        delta=margin_data_diff)
            # Bar Graph
            fig = px.bar(df_margin,
                            x=df_margin.columns[0],
                            y=df_margin.columns[1],
                            # color=df_margin.columns[1]
                            )
            st.plotly_chart(fig, width=0, height=300, use_container_width=True)

        with st.container(border=True, height=400):
            open_PO_result = run_sql_query(open_PO_qry)
            open_PO_df = pd.DataFrame(open_PO_result)
            open_PO_df.columns = open_PO_df.columns.str.replace('_', ' ')
            open_PO_headers = open_PO_df.columns
            st.subheader("Open Purchase Orders:", divider='rainbow')
            st.markdown(
                tabulate(open_PO_df, tablefmt="html", headers=open_PO_headers, floatfmt=".2f",
                            showindex=False),
                unsafe_allow_html=True)

        with st.container(border=True, height=400):
            invoices_WIP_result = run_sql_query(invoices_WIP_qry)
            invoices_WIP_df = pd.DataFrame(invoices_WIP_result)
            invoices_WIP_df.columns = invoices_WIP_df.columns.str.replace('_', ' ')
            invoices_WIP_headers = invoices_WIP_df.columns
            st.subheader("Invoices Work in Progress:", divider='rainbow')
            st.markdown(
                tabulate(invoices_WIP_df, tablefmt="html", headers=invoices_WIP_headers, floatfmt=".2f",
                            showindex=False),
                unsafe_allow_html=True)

        with st.container(border=True, height=400):
            unbilled_revenue_result = run_sql_query(unbilled_revenue_qry)
            unbilled_revenue_df = pd.DataFrame(unbilled_revenue_result)
            unbilled_revenue_df.columns = unbilled_revenue_df.columns.str.replace('_', ' ')
            unbilled_revenue_headers = unbilled_revenue_df.columns
            st.subheader("Unbilled Revenue:", divider='rainbow')
            st.markdown(
                tabulate(unbilled_revenue_df, tablefmt="html", headers=unbilled_revenue_headers,
                            floatfmt=".2f", showindex=False),
                unsafe_allow_html=True)

    elif select_source == 'Query Financial Data':
        ### Setup the Home Page
        str_input = st.chat_input("Enter your question:")
        st.markdown("<h2>AI Assistant :</h2>", unsafe_allow_html=True)
        st.markdown("""Welcome! I am Finance Assistant of your company. 
                    I possess the ability to extract information from your company's financial statements like expense, invoice, balance sheet etc. 
                    Please ask me questions and I will try my level best to provide accurate responses.""")
        ### Add a select box to add scope to choose the model
        llm_selected = st.selectbox("Choose a model :",
                                    options=["OpenAI - Gpt 4.0 Turbo", "OpenAI - Gpt 4.o", "OpenAI - Gpt 4.0",
                                                "OpenAI - Gpt 3.5 Turbo"])
        if llm_selected == "OpenAI - Gpt 4.0 Turbo":
            llm_model_name = "gpt-4"
        elif llm_selected == "OpenAI - Gpt 4.o":
            llm_model_name = "gpt-4"
        elif llm_selected == "OpenAI - Gpt 4.0":
            llm_model_name = "gpt-4"
        elif llm_selected == "OpenAI - Gpt 3.5 Turbo":
            llm_model_name = "gpt-3.5-turbo"
        else:
            llm_model_name = "gpt-4"

        ### Save the User Chat History
        new_data = {"User_Chat_History": str_input}
        chat_df_1 = chat_df_1._append(new_data, ignore_index=True)
        chat_df_1 = chat_df_1.dropna().drop_duplicates()
        chat_df_1 = chat_df_1.sort_index(axis=0, ascending=False)
        chat_df_1.to_csv(his_file_1, index=False)

        ### Add a button to reset the User Chat History
        chat_reset = st.sidebar.button(":orange[Clear Chat History]", type="secondary",
                                        key="Clear_Chat_History")
        if chat_reset:
            chat_df = pd.DataFrame(columns=["User_Chat_History"])
            chat_df.to_csv(his_file_1, index=False)

        ### Give flexibility to the User to Select from the Chat History
        with st.sidebar:
            for index, row in chat_df_1.iterrows():
                if st.button(f"{row['User_Chat_History']}"):
                    str_input = str(row['User_Chat_History'])

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                role = message["role"]
                df_str = message["content"]
                if role == "user":
                    st.markdown(df_str, unsafe_allow_html=True)
                    continue
                if df_str.find("<separator>") > -1:
                    csv_str = df_str[:df_str.index("<separator>")]
                    analysis_str = df_str[df_str.index("<separator>") + len("<separator>"):]
                    csv = StringIO(csv_str)
                    df_data = pd.read_csv(csv, sep=',')
                    df_data.columns = df_data.columns.str.replace('_', ' ')
                    headers = df_data.columns
                    st.markdown(f'<p style="font-family:sans-serif; font-size:15px">{analysis_str}</p>',
                                unsafe_allow_html=True)
                    if len(df_data.index) >= 2 and len(df_data.columns) >= 2 and len(df_data.columns) <= 3:
                        with st.expander("Graph:"):
                            plot_chart(df_data)
                    with st.expander("Table Output:"):
                        st.markdown(
                            tabulate(df_data, tablefmt="html", headers=headers, floatfmt="0f",
                                        showindex=False),
                            unsafe_allow_html=True)
                else:
                    st.markdown(df_str)

        # This is the new block for the 'Query Financial Data' section
    if prompt := str_input("Enter your question:"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
    
        with st.chat_message("assistant"):
            try:
                with st.spinner("Running UiPath Robot..."):
                    # This function call has been changed
                    robot_response_payload = uipath_connector.run_robot_and_get_output(prompt)
                
                if robot_response_payload and 'data' in robot_response_payload:
                    df = pd.read_json(robot_response_payload['data'], orient='split')
                else:
                    error_message = robot_response_payload.get("error", "No data returned from robot.")
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    st.stop()

                df_analysis = str(df)
                sql_result_analysis = result_analysis(df_analysis, prompt)

                st.markdown(f'<p style="font-family:sans-serif; font-size:15px">{sql_result_analysis}</p>', unsafe_allow_html=True)
                
                if not df.empty:
                    with st.expander("Table Output:"):
                        st.markdown(tabulate(df, tablefmt="html", headers=df.columns.str.replace('_', ' '), floatfmt=".2f", showindex=False), unsafe_allow_html=True)
                
                out_data = df.to_csv(sep=',', index=False) + "<separator>" + sql_result_analysis
                st.session_state.messages.append({"role": "assistant", "content": out_data})

            except Exception as e:
                error_str = f"An error occurred while running the UiPath Robot: {e}"
                st.error(error_str)
                st.session_state.messages.append({"role": "assistant", "content": error_str})

    elif select_source == 'Business Insights':
        st.switch_page("pages/Dashboard.py")
        #st.switch_page("1_(house)_Dashboard.py")
    
    elif select_source == 'Query Month End Reports':
        str_input = st.chat_input("Enter your question:")
        st.markdown("<h2>AI Assistant :</h2>", unsafe_allow_html=True)
        st.markdown("""Welcome! I'm your AI assistant specialized to extract information from exception report insights. 
                        My purpose is to assist with any queries related to exception reports within your organization.
                        Please enter your questions in the text box below, or you can choose from the list provided on the left panel.
                        I'm here to provide accurate responses to your inquiries.""")
        ### Add a select box to add scope to choose the model
        llm_selected = st.selectbox("Choose a model :",
                                    options=["OpenAI - Gpt 4.0 Turbo", "OpenAI - Gpt 4.o", "OpenAI - Gpt 4.0",
                                                "OpenAI - Gpt 3.5 Turbo"])
        if llm_selected == "OpenAI - Gpt 4.0 Turbo":
            llm_model_name = "gpt-4"
        elif llm_selected == "OpenAI - Gpt 4.o":
            llm_model_name = "gpt-4"
        elif llm_selected == "OpenAI - Gpt 4.0":
            llm_model_name = "gpt-4"
        elif llm_selected == "OpenAI - Gpt 3.5 Turbo":
            llm_model_name = "gpt-3.5-turbo"
        else:
            llm_model_name = "gpt-4"

        new_data = {"User_Chat_History": str_input}
        chat_df_2 = chat_df_2._append(new_data, ignore_index=True)
        chat_df_2 = chat_df_2.dropna().drop_duplicates()
        chat_df_2 = chat_df_2.sort_index(axis=0, ascending=False)
        chat_df_2.to_csv(his_file_2, index=False)
        chat_reset = st.sidebar.button(":orange[Clear Chat History]", type="secondary",
                                        key="Clear_Chat_History")
        if chat_reset:
            chat_df_2 = pd.DataFrame(columns=["User_Chat_History"])
            chat_df_2.to_csv(his_file_2, index=False)

        with st.sidebar:
            for index, row in chat_df_2.iterrows():
                if st.button(f"{row['User_Chat_History']}"):
                    str_input = str(row['User_Chat_History'])

        for message in st.session_state.messages1:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)

        if prompt1 := str_input:
            st.chat_message("user").markdown(prompt1, unsafe_allow_html=True)
            st.session_state.messages1.append({"role": "user", "content": prompt1})
            with st.chat_message("assistant"):
                result = prompts.letter_chain(str_input)
                answer = result['result']
                st.markdown(answer)
                st.session_state.messages1.append({"role": "assistant", "content": answer})

    elif select_source == 'Trigger Month End':
        st.markdown("<h2>AI Assistant :</h2>", unsafe_allow_html=True)
        st.markdown("""Welcome! I'm your AI assistant. 
                    My purpose is to start the AP month end process and check the status for you.
                    Please click on the below button, I will trigger the process for you.""")

        # Workflow buttons
        # Create a horizontal layout with columns
        col1, col2, col3 = st.columns(3)
        # First horizontal layout
        with col1:
            # Create a container for the buttons
            with st.container(border=True, height=600):
                st.subheader("PeriodClose -3", divider='rainbow')
                # First Button
                st.markdown(":grey-background[**Review the Tax Forecast**]")
                if st.session_state.button13 == 1 or st.session_state.master_button == 0:
                    st.success('The Process has completed successfully!', icon="✅")
                    st.toggle("Mark the Process as Complete", key="13_1", value=1, disabled=True)
                else:
                    my_bar13 = st.info("Process yet to be Start", icon="ℹ️")
                    if st.toggle("Mark the Process as Complete", key="13_2"):
                        st.session_state.button13 = 1
                        st.session_state.OverallProcess += 8.33
                        st.session_state.TAX_Process += 100
                        my_bar13.success('The Process has completed successfully!', icon="✅")
                st.markdown(":grey-background[**Create Accounting**]")
                if st.session_state.button1 == 1 or st.session_state.master_button == 0:
                    st.success('The Process has completed successfully!', icon="✅")
                    st.button("RUN 🤖", key="Create_Accounting_RERUN", disabled=True)
                else:
                    my_bar1 = st.info("Process yet to be Start", icon="ℹ️")
                    if st.button("RUN 🤖", key="Create_Accounting_RUN"):
                        st.session_state.button1 = 1
                        UiPath_API_Queue_Load.add_data_to_queue('Create Accounting')
                        st.session_state.OverallProcess += 6.25
                        st.session_state.AP_Process += 20
                        for percent_complete in range(100):
                            queue_item_status, queue_item_progress = UiPath_API_Queue_Load.read_status_in_queue()
                            if queue_item_status in ('New', 'InProgress'):
                                my_bar1.progress(percent_complete + 1,
                                                    text="Operation is in progress. Please wait for sometime...")
                                time.sleep(2)
                            elif queue_item_status == 'Successful':
                                my_bar1.success('The Process has completed successfully!', icon="✅")
                                break
                            else:
                                my_bar1.error('Something went wrong. Please check the process', icon="⚠️")
                                break
                st.markdown(":grey-background[**AP Accurals**]")
                if st.session_state.button16 == 1 or st.session_state.master_button == 0:
                    st.success('The Process has completed successfully!', icon="✅")
                    st.button("RUN 🤖", key="Create_AP_Accurals_RERUN", disabled=True)
                else:
                    my_bar16 = st.info("Process yet to be Start", icon="ℹ️")
                    if st.button("RUN 🤖", key="Create_AP_Accurals_RUN"):
                        st.session_state.button16 = 1
                        st.session_state.OverallProcess += 6.25
                        st.session_state.AP_Process += 20
                        url = "http://ec2-18-225-92-148.us-east-2.compute.amazonaws.com:9090/ws/simple/executeJournal"
                        response = requests.post(
                            url,
                            auth=HTTPBasicAuth(api_username, api_password),
                            verify=False,  # Skip SSL cert verification if self-signed
                            timeout=10
                        )

                        for percent_complete in range(100):
                            my_bar16.progress(percent_complete + 10,
                                                    text="Operation is in progress. Please wait for sometime...")
                            time.sleep(2)
                            if response.status_code == 200:
                                my_bar16.success('The Process has completed successfully!', icon="✅")
                                break

            with st.container(border=True, height=600):
                st.subheader("PeriodClose +1", divider='rainbow')
                # First Button
                st.markdown(":grey-background[**Update Margin analysis for Tax provision**]")
                if st.session_state.button15 == 1 or st.session_state.master_button == 0:
                    st.success('The Process has completed successfully!', icon="✅")
                    st.toggle("Mark the Process as Complete", key="15_1", value=1, disabled=True)
                else:
                    my_bar15 = st.info("Process yet to be Start", icon="ℹ️")
                    if st.toggle("Mark the Process as Complete", key="15_2"):
                        st.session_state.button15 = 1
                        my_bar15.success('The Process has completed successfully!', icon="✅")
                st.markdown(":grey-background[**GL Transfer**]")
                if st.session_state.button2 == 1 or st.session_state.master_button == 0:
                    st.success('The Process has completed successfully!', icon="✅")
                    st.button("RUN 🤖", key="GL_Transfer_RERUN", disabled=True)
                else:
                    my_bar2 = st.info("Process yet to be Start", icon="ℹ️")
                    if st.button("RUN 🤖", key="GL_Transfer_RUN"):
                        st.session_state.button2 = 1
                        UiPath_API_Queue_Load.add_data_to_queue('GL Transfer')
                        st.session_state.OverallProcess += 6.25
                        st.session_state.AR_Process += 25
                        for percent_complete in range(100):
                            queue_item_status, queue_item_progress = UiPath_API_Queue_Load.read_status_in_queue()
                            if queue_item_status in ('New', 'InProgress'):
                                my_bar2.progress(percent_complete + 1,
                                                    text="Operation is in progress. Please wait for sometime...")
                                time.sleep(2)
                            elif queue_item_status == 'Successful':
                                my_bar2.success('The Process has completed successfully!', icon="✅")
                                break
                            else:
                                my_bar2.error('Something went wrong. Please check the process', icon="⚠️")
                                break

                # Second Button
                st.markdown(":grey-background[**Trial Balance Report**]")
                if st.session_state.button3 == 1 or st.session_state.master_button == 0:
                    st.success('The Process has completed successfully!', icon="✅")
                    st.button("RUN 🤖", key="Trial_Balance_Report_RERUN", disabled=True)
                else:
                    my_bar3 = st.info("Process yet to be Start", icon="ℹ️")
                    if st.button("RUN 🤖", key="Trial_Balance_Report_RUN"):
                        st.session_state.button3 = 1
                        UiPath_API_Queue_Load.add_data_to_queue('Trial Balance Report')
                        st.session_state.OverallProcess += 6.25
                        st.session_state.AP_Process += 20
                        for percent_complete in range(100):
                            queue_item_status, queue_item_progress = UiPath_API_Queue_Load.read_status_in_queue()
                            if queue_item_status in ('New', 'InProgress'):
                                my_bar3.progress(percent_complete + 1,
                                                    text="Operation is in progress. Please wait for sometime...")
                                time.sleep(2)
                            elif queue_item_status == 'Successful':
                                my_bar3.success('The Process has completed successfully!', icon="✅")
                                break
                            else:
                                my_bar3.error('Something went wrong. Please check the process', icon="⚠️")
                                break

        # Second horizontal layout
        with col2:
            # Create a container for the buttons
            with st.container(border=True, height=600):
                st.subheader("PeriodClose -2", divider='rainbow')
                # First Button
                st.markdown(":grey-background[**Revenue and Margin forecasted entries in Adaptive**]")
                if st.session_state.button14 == 1 or st.session_state.master_button == 0:
                    st.success('The Process has completed successfully!', icon="✅")
                    st.toggle("Mark the Process as Complete", key="14_1", value=1, disabled=True)
                else:
                    my_bar14 = st.info("Process yet to be Start", icon="ℹ️")
                    if st.toggle("Mark the Process as Complete", key="14_2"):
                        st.session_state.button14 = 1
                        my_bar14.success('The Process has completed successfully!', icon="✅")
                st.markdown(":grey-background[**Accounting Reconciliation**]")
                if st.session_state.button4 == 1 or st.session_state.master_button == 0:
                    st.success('The Process has completed successfully!', icon="✅")
                    st.button("RUN 🤖", key="Accounting_Reconciliation_RERUN", disabled=True)
                else:
                    my_bar4 = st.info("Process yet to be Start", icon="ℹ️")
                    if st.button("RUN 🤖", key="Accounting_Reconciliation_RUN"):
                        st.session_state.button4 = 1
                        UiPath_API_Queue_Load.add_data_to_queue('Reconciliation')
                        st.session_state.OverallProcess += 6.25
                        st.session_state.GL_Process += 50
                        for percent_complete in range(100):
                            queue_item_status, queue_item_progress = UiPath_API_Queue_Load.read_status_in_queue()
                            if queue_item_status in ('New', 'InProgress'):
                                my_bar4.progress(percent_complete + 1,
                                                    text="Operation is in progress. Please wait for sometime...")
                                time.sleep(2)
                            elif queue_item_status == 'Successful':
                                my_bar4.success('The Process has completed successfully!', icon="✅")
                                break
                            else:
                                my_bar4.error('Something went wrong. Please check the process', icon="⚠️")
                                break
                # Second Button
                st.markdown(":grey-background[**IT Accrual check**]")
                if st.session_state.button5 == 1 or st.session_state.master_button == 0:
                    st.success('The Process has completed successfully!', icon="✅")
                    st.toggle("Mark the Process as Complete", key="5_1", value=1, disabled=True)
                else:
                    my_bar5 = st.info("Process yet to be Start", icon="ℹ️")
                    if st.toggle("Mark the Process as Complete", key="5_2"):
                        st.session_state.button5 = 1
                        st.session_state.OverallProcess += 6.25
                        st.session_state.AR_Process += 25
                        my_bar5.success('The Process has completed successfully!', icon="✅")

            with st.container(border=True, height=600):
                st.subheader("PeriodClose +2", divider='rainbow')
                # First Button
                st.markdown(":grey-background[**Inventory Recon**]")
                if st.session_state.button6 == 1 or st.session_state.master_button == 0:
                    st.success('The Process has completed successfully!', icon="✅")
                    st.toggle("Mark the Process as Complete", key="6_1", value=1, disabled=True)
                else:
                    my_bar6 = st.info("Process yet to be Start", icon="ℹ️")
                    if st.toggle("Mark the Process as Complete", key="6_2"):
                        st.session_state.button6 = 1
                        st.session_state.OverallProcess += 6.25
                        st.session_state.AR_Process += 25
                        my_bar6.success('The Process has completed successfully!', icon="✅")

                # Second Button
                st.markdown(":grey-background[**Invoice Aging Check**]")
                if st.session_state.button7 == 1 or st.session_state.master_button == 0:
                    st.success('The Process has completed successfully!', icon="✅")
                    st.button("RUN 🤖", key="Invoice_Aging_Check_RERUN", disabled=True)
                else:
                    my_bar7 = st.info("Process yet to be Start", icon="ℹ️")
                    if st.button("RUN 🤖", key="Invoice_Aging_Check_RUN"):
                        st.session_state.button7 = 1
                        UiPath_API_Queue_Load.add_data_to_queue('Invoice Aging')
                        st.session_state.OverallProcess += 6.25
                        st.session_state.GL_Process += 50
                        for percent_complete in range(100):
                            queue_item_status, queue_item_progress = UiPath_API_Queue_Load.read_status_in_queue()
                            if queue_item_status in ('New', 'InProgress'):
                                my_bar7.progress(percent_complete + 1,
                                                    text="Operation is in progress. Please wait for sometime...")
                                time.sleep(2)
                            elif queue_item_status == 'Successful':
                                my_bar7.success('The Process has completed successfully!', icon="✅")
                                break
                            else:
                                my_bar7.error('Something went wrong. Please check the process', icon="⚠️")
                                break

                # Third Button
                st.markdown(":grey-background[**Tax and Treasury Analysis**]")
                if st.session_state.button8 == 1 or st.session_state.master_button == 0:
                    st.success('The Process has completed successfully!', icon="✅")
                    st.toggle("Mark the Process as Complete", key="8_1", value=1, disabled=True)
                else:
                    my_bar8 = st.info("Process yet to be Start", icon="ℹ️")
                    if st.toggle("Mark the Process as Complete", key="8_2"):
                        st.session_state.button8 = 1
                        st.session_state.OverallProcess += 6.25
                        st.session_state.AR_Process += 25
                        my_bar8.success('The Process has completed successfully!', icon="✅")

            # Third horizontal layout
            with col3:
                with st.container(border=True, height=600):
                    st.subheader("PeriodClose -1", divider='rainbow')
                    # First Button
                    st.markdown(":grey-background[**Open PO and GL period**]")
                    if st.session_state.button9 == 1 or st.session_state.master_button == 0:
                        st.success('The Process has completed successfully!', icon="✅")
                        st.toggle("Mark the Process as Complete", key="9_1", value=1, disabled=True)
                    else:
                        my_bar9 = st.info("Process yet to be Start", icon="ℹ️")
                        if st.toggle("Mark the Process as Complete", key="9_2"):
                            st.session_state.button9 = 1
                            st.session_state.OverallProcess += 6.25
                            st.session_state.INV_Process += 100
                            my_bar9.success('The Process has completed successfully!', icon="✅")

                    # Second Button
                    st.markdown(":grey-background[**Unaccounted transaction check**]")
                    if st.session_state.button10 == 1 or st.session_state.master_button == 0:
                        st.success('The Process has completed successfully!', icon="✅")
                        st.button("RUN 🤖", key="Unaccounted_transaction_check_RERUN", disabled=True)
                    else:
                        my_bar10 = st.info("Process yet to be Start", icon="ℹ️")
                        if st.button("RUN 🤖", key="Unaccounted_transaction_check_RUN"):
                            st.session_state.button10 = 1
                            UiPath_API_Queue_Load.add_data_to_queue('Unaccounted Transaction Report')
                            st.session_state.OverallProcess += 6.25
                            st.session_state.AP_Process += 20
                            for percent_complete in range(100):
                                queue_item_status, queue_item_progress = UiPath_API_Queue_Load.read_status_in_queue()
                                if queue_item_status in ('New', 'InProgress'):
                                    my_bar10.progress(percent_complete + 1,
                                                        text="Operation is in progress. Please wait for sometime...")
                                    time.sleep(2)
                                elif queue_item_status == 'Successful':
                                    my_bar10.success('The Process has completed successfully!', icon="✅")
                                    break
                                else:
                                    my_bar10.error('Something went wrong. Please check the process', icon="⚠️")
                                    break

                    # Third Button
                    st.markdown(":grey-background[**Exception Correction**]")
                    if st.session_state.button11 == 1 or st.session_state.master_button == 0:
                        st.success('The Process has completed successfully!', icon="✅")
                        st.button("RUN 🤖", key="Exception_Correction_RERUN", disabled=True)
                    else:
                        my_bar11 = st.info("Process yet to be Start", icon="ℹ️")
                        if st.button("RUN 🤖", key="Exception_Correction_RUN"):
                            st.session_state.button11 = 1
                            UiPath_API_Queue_Load.add_data_to_queue('Exception Processing')
                            st.session_state.OverallProcess += 6.25
                            st.session_state.AR_Process += 25
                            for percent_complete in range(100):
                                queue_item_status, queue_item_progress = UiPath_API_Queue_Load.read_status_in_queue()
                                if queue_item_status in ('New', 'InProgress'):
                                    my_bar11.progress(percent_complete + 1,
                                                        text="Operation is in progress. Please wait for sometime...")
                                    time.sleep(2)
                                elif queue_item_status == 'Successful':
                                    my_bar11.success('The Process has completed successfully!', icon="✅")
                                    break
                                else:
                                    my_bar11.error('Something went wrong. Please check the process', icon="⚠️")
                                    break

                with st.container(border=True, height=600):
                    st.subheader("PeriodClose +3", divider='rainbow')
                    # First Button
                    st.markdown(":grey-background[**Close Period**]")
                    if st.session_state.button12 == 1 or st.session_state.master_button == 0:
                        st.success('The Process has completed successfully!', icon="✅")
                        st.toggle("Mark the Process as Complete", key="12_1", value=1, disabled=True)
                    else:
                        my_bar12 = st.info("Process yet to be Start", icon="ℹ️")
                        if st.toggle("Mark the Process as Complete", key="12_2"):
                            st.session_state.button12 = 1
                            st.session_state.OverallProcess += 6.25
                            st.session_state.AP_Process += 20
                            my_bar12.success('The Process has completed successfully!', icon="✅")



except Exception as err:
    with st.chat_message("assistant"):
        if "messages" not in st.session_state.keys():
            st.session_state.messages = []
        err_msg = "Something Went Wrong - " + str(err)
        st.markdown(err_msg)
        st.session_state.messages.append({"role": "assistant", "content": err_msg})
