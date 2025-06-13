import streamlit as st
import joblib
import numpy as np
import pandas as pd
import bz2
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Loan Default Prediction Tool",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Override with simpler theme
st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF;
    }
    .stButton > button {
        background-color: #2E86AB;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load dataset
file_url = "https://raw.githubusercontent.com/sharmaraghav644/LoanDefaultPredictionApp/refs/heads/main/Loan_default_predication_kaggle.csv"
df = pd.read_csv(file_url)

# RAG Knowledge Base - Based on your document sources with links
LOAN_KNOWLEDGE_BASE = {
    "credit_score": {
        "content": """
        Your credit score is like a financial report card that shows how well you've managed borrowed money in the past. 
        It ranges from 300 to 850, with higher scores being better. Banks use this number to decide if they should lend you money 
        and what interest rate to charge. A good credit score (700+) means you're likely to pay back loans on time, 
        so banks offer you better deals with lower interest rates.
        """,
        "sources": [
            {"title": "How Your Credit Score Impacts Your Financial Future", "url": "https://www.finra.org/investors/personal-finance/how-your-credit-score-impacts-your-financial-future"},
            {"title": "How Credit Score Impacts You", "url": "https://www.ratehub.ca/credit-score"}
        ]
    },
    
    "income": {
        "content": """
        Your income shows banks how much money you earn regularly, which helps them figure out if you can afford 
        to pay back a loan. Higher income generally means you're more likely to handle loan payments without problems. 
        Banks don't just look at how much you make, but also how stable your income is - steady paychecks are preferred 
        over unpredictable earnings.
        """,
        "sources": [
            {"title": "Does your income impact your credit score?", "url": "https://creditblog.capitalone.ca/latest-stories/income-credit-score"}
        ]
    },
    
    "loan_amount": {
        "content": """
        The amount of money you want to borrow matters a lot. If you ask for too much money compared to what you earn, 
        banks get worried that you might struggle to pay it back. Think of it like this: if you make $50,000 a year
        but want to borrow $200,000, that's a red flag. Banks prefer when the loan amount makes sense compared to your income.
        """,
        "sources": [
            {"title": "Maximum Loan Amount: Definition and Factors Lenders Consider", "url": "https://www.investopedia.com/terms/m/maximum_loan_amount.asp"}
        ]
    },
    
    "dti_ratio": {
        "content": """
        Debt-to-Income ratio (DTI) is simply how much of your monthly income goes toward paying debts. 
        For example, if you earn $5,000 per month and pay $1,500 in various debt payments, your DTI is 30%. 
        Banks like to see this number below 36% because it means you're not drowning in debt payments and 
        can handle a new loan without financial stress.
        """,
        "sources": [
            {"title": "Debt-to-Income (DTI) Ratio: What's Good and...", "url": "https://www.investopedia.com/terms/d/dti.asp"},
            {"title": "High Debt-to-Income Ratio: Impacts & Loan Options", "url": "https://griffinfunding.com/blog/mortgage/what-happens-if-you-have-a-high-dti/"}
        ]
    },
    
    "interest_rate": {
        "content": """
        Interest rate is the extra money you pay for borrowing. Think of it as the 'rental fee' for using the bank's money. 
        Higher interest rates make your monthly payments bigger and the total loan more expensive. When interest rates 
        go up significantly, some people struggle to keep up with payments, which increases the chance they might default 
        on their loan.
        """,
        "sources": [
            {"title": "Understanding The Connection Between Loan Term and Interest Rate", "url": "https://www.amres.com/amres-resources/understanding-the-connection-between-loan-term-and-interest-rate"},
            {"title": "Default and Interest Rate Shocks: Renegotiation Matters", "url": "https://www.gc.cuny.edu/sites/default/files/2021-07/AEKNdraft2021.pdf"}
        ]
    },
    
    "employment_length": {
        "content": """
        How long you've been at your current job shows banks whether you have stable income. 
        If you've been working at the same place for 2+ years, banks see this as a good sign that you're likely 
        to keep earning money consistently. Job-hoppers or people who just started new jobs are seen as riskier 
        because their income might not be as predictable.
        """,
        "sources": [
            {"title": "How important is length of time at current job to get approved", "url": "https://www.reddit.com/r/FirstTimeHomeBuyer/comments/17hpso5/how_important_is_length_of_time_at_current_job_to/"}
        ]
    },
    
    "loan_term": {
        "content": """
        Loan term is simply how long you have to pay back the money. Longer terms mean smaller monthly payments 
        but you'll pay more interest over time. Shorter terms mean bigger monthly payments but less total interest. 
        Banks consider both your ability to handle the monthly payment and the total risk over the loan's lifetime.
        """,
        "sources": [
            {"title": "Understanding The Connection Between Loan Term and Interest Rate", "url": "https://www.amres.com/amres-resources/understanding-the-connection-between-loan-term-and-interest-rate"}
        ]
    },
    
    "education": {
        "content": """
        Your education level gives banks a hint about your earning potential and financial knowledge. 
        People with higher education (college degrees, master's, PhD) statistically have lower default rates 
        because they typically earn more money over their careers and understand financial responsibilities better. 
        It's not a guarantee, but it's a helpful indicator for banks.
        """,
        "sources": []
    },
    
    "marital_status": {
        "content": """
        Being married, single, or divorced can affect your loan application, especially for joint applications. 
        Married couples might have combined income which can help qualification, but they also have combined debts. 
        Banks look at the overall financial picture - sometimes having a spouse helps, sometimes it doesn't, 
        depending on both people's financial situations.
        """,
        "sources": [
            {"title": "How Marital Status Affects Credit Card and Loan Applications", "url": "https://www.rocketlawyer.com/family-and-personal/family-matters/marriage/legal-guide/how-marital-status-affects-credit-card-and-loan-applications"}
        ]
    },
    
    "loan_purpose": {
        "content": """
        Why you need the money matters to banks. Some purposes are considered safer than others. 
        For example, a home loan is backed by the house itself, so it's less risky for banks. 
        Business loans might be riskier because businesses can fail. Auto loans are in the middle 
        because cars can be repossessed if needed. Banks adjust their decision based on what you plan to do with the money.
        """,
        "sources": [
            {"title": "Loan - Wikipedia", "url": "https://en.wikipedia.org/wiki/Loan"}
        ]
    },
    
    "has_co_signer": {
        "content": """
        A co-signer is someone who promises to pay your loan if you can't. Having a co-signer with good credit 
        significantly reduces the bank's risk because there are now two people responsible for the debt instead of one. 
        This often leads to better loan terms, lower interest rates, or approval for people who might not qualify alone.
        """,
        "sources": []
    },
    
    "has_mortgage": {
        "content": """
        Already having a mortgage shows banks that you've successfully managed a large, long-term debt before. 
        This can be seen as positive (you're experienced with big loans) or negative (you already have a major financial obligation). 
        It depends on your overall financial picture - if you're handling your mortgage well and have good income, 
        it can actually help your application.
        """,
        "sources": [
            {"title": "What Affects Your Ability to Get a Loan?", "url": "https://www.incharge.org/blog/what-affects-your-ability-to-get-a-home-loan/"},
            {"title": "Home Equity Line of Credit or Loan", "url": "https://www.rbcroyalbank.com/mortgages/using-home-equity.html"}
        ]
    },
    
    "has_dependents": {
        "content": """
        Having dependents (children or family members you financially support) means you have additional monthly expenses 
        that banks need to consider. While this doesn't automatically disqualify you, banks factor in these extra costs 
        when calculating whether you can afford a new loan payment. More dependents mean higher living expenses, 
        which could affect your ability to repay.
        """,
        "sources": []
    }
}

# Simplified fallback explanation function (no GPT-2 required)
def generate_simple_explanation(prediction_probability, user_inputs):
    """Generate a simple explanation without requiring GPT-2"""
    
    risk_percentage = prediction_probability * 100
    
    # Analyze key factors
    factors = []
    
    # Credit Score Analysis
    credit_score = user_inputs.get('credit_score', 700)
    if credit_score >= 750:
        factors.append("Your excellent credit score strongly supports loan approval")
    elif credit_score >= 700:
        factors.append("Your good credit score is favorable for loan approval")
    elif credit_score >= 650:
        factors.append("Your fair credit score may require additional review")
    else:
        factors.append("Your credit score presents some challenges for loan approval")
    
    # Income vs Loan Amount
    income = user_inputs.get('income', 50000)
    loan_amount = user_inputs.get('loan_amount', 10000)
    loan_to_income = loan_amount / income if income > 0 else 0
    
    if loan_to_income <= 0.2:
        factors.append("The loan amount is very reasonable compared to your income")
    elif loan_to_income <= 0.5:
        factors.append("The loan amount is manageable with your current income")
    else:
        factors.append("The loan amount is quite high relative to your income, which increases risk")
    
    # DTI Analysis
    dti_ratio = user_inputs.get('dti_ratio', 0.3)
    if dti_ratio <= 0.28:
        factors.append("Your debt-to-income ratio is excellent")
    elif dti_ratio <= 0.36:
        factors.append("Your debt-to-income ratio is acceptable")
    else:
        factors.append("Your debt-to-income ratio is concerning and may affect approval")
    
    # Employment Stability
    months_employed = user_inputs.get('months_employed', 60)
    if months_employed >= 24:
        factors.append("Your employment history shows good stability")
    else:
        factors.append("Your shorter employment history may be a concern")
    
    # Co-signer benefit
    if user_inputs.get('has_co_signer') == "Yes":
        factors.append("Having a co-signer significantly improves your application")
    
    # Create explanation based on risk level
    if risk_percentage <= 30:
        explanation = f"With a {risk_percentage:.1f}% default risk, this is a strong loan application. "
        explanation += "Key strengths: " + "; ".join(factors[:3]) + "."
    elif risk_percentage <= 60:
        explanation = f"With a {risk_percentage:.1f}% default risk, this application has mixed factors. "
        explanation += "Main considerations: " + "; ".join(factors[:3]) + "."
    else:
        explanation = f"With a {risk_percentage:.1f}% default risk, this application faces significant challenges. "
        explanation += "Key concerns: " + "; ".join(factors[:3]) + "."
    
    return explanation

# Improved GPT-2 model loading with better error handling
@st.cache_resource
def load_gpt2_model():
    """Load and cache GPT-2 model and tokenizer with improved error handling"""
    try:
        # Try gpt2 (smaller model) first for better compatibility
        model_name = "gpt2"  # Changed from gpt2-medium to gpt2
        
        st.info("🤖 Loading AI model... This may take a moment on first run.")
        
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add padding token
        tokenizer.pad_token = tokenizer.eos_token
        
        # Set model to evaluation mode
        model.eval()
        
        st.success("✅ AI model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        st.warning(f"⚠️ Could not load AI model: {str(e)}")
        st.info("Using simplified explanation instead.")
        return None, None

# Function to load non-DL models safely
def load_bz2_model(file_path):
    try:
        with bz2.BZ2File(file_path, "rb") as f:
            return joblib.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to load the selected model dynamically
def get_model(choice):
    if choice == "XGBoost":
        return load_bz2_model("xgb_model_compressed.pkl.bz2")
    elif choice == "Random Forest":
        return load_bz2_model("rf_model_compressed.pkl.bz2")

# Updated RAG retrieval function with better contextual filtering
def retrieve_relevant_sources(user_inputs, prediction_probability):
    """Retrieve relevant knowledge sources based on user inputs and prediction"""
    relevant_sources = []
    
    # Risk-based prioritization
    is_high_risk = prediction_probability > 0.5
    
    # ALWAYS INCLUDE - Core factors that affect all loans
    relevant_sources.append(("Credit Score", LOAN_KNOWLEDGE_BASE["credit_score"]))
    relevant_sources.append(("Income", LOAN_KNOWLEDGE_BASE["income"]))
    
    # CONDITIONAL INCLUSION - Only show if problematic or relevant
    
    # DTI Issues - Only show if concerning
    if user_inputs.get('dti_ratio', 0) > 0.36:
        relevant_sources.append(("Debt-to-Income Ratio", LOAN_KNOWLEDGE_BASE["dti_ratio"]))
    
    # High Interest Rate - Only show if rate is high
    if user_inputs.get('interest_rate', 0) > 12:
        relevant_sources.append(("Interest Rate", LOAN_KNOWLEDGE_BASE["interest_rate"]))
    
    # Employment Concerns - Only show if employment is short
    if user_inputs.get('months_employed', 60) < 24:  # Less than 2 years
        relevant_sources.append(("Employment Length", LOAN_KNOWLEDGE_BASE["employment_length"]))
    
    # Loan Amount vs Income ratio - Only show if concerning
    loan_to_income_ratio = user_inputs.get('loan_amount', 0) / max(user_inputs.get('income', 1), 1)
    if loan_to_income_ratio > 0.5:  # Loan is more than 50% of annual income
        relevant_sources.append(("Loan Amount", LOAN_KNOWLEDGE_BASE["loan_amount"]))
    
    # Education - Only show if low education AND high risk
    if user_inputs.get('education') == "High School" and is_high_risk:
        relevant_sources.append(("Education Level", LOAN_KNOWLEDGE_BASE["education"]))
    
    # Positive factors - Show these if they help the case
    if user_inputs.get('has_co_signer') == "Yes":
        relevant_sources.append(("Co-Signer Advantage", LOAN_KNOWLEDGE_BASE["has_co_signer"]))
    
    # Mortgage - Only relevant if they have one (affects DTI and shows payment history)
    if user_inputs.get('has_mortgage') == "Yes":
        relevant_sources.append(("Existing Mortgage Impact", LOAN_KNOWLEDGE_BASE["has_mortgage"]))
    
    # Dependents - Only show if they have dependents (affects expenses)
    if user_inputs.get('has_dependents') == "Yes":
        relevant_sources.append(("Financial Dependents", LOAN_KNOWLEDGE_BASE["has_dependents"]))
    
    # Loan Purpose - Only show for riskier purposes or if high risk
    risky_purposes = ["Business", "Other"]
    if user_inputs.get('loan_purpose') in risky_purposes or is_high_risk:
        relevant_sources.append(("Loan Purpose Impact", LOAN_KNOWLEDGE_BASE["loan_purpose"]))
    
    # Marital Status - Only relevant for joint applications or if it affects DTI
    if user_inputs.get('marital_status') == "Married":
        relevant_sources.append(("Marital Status Considerations", LOAN_KNOWLEDGE_BASE["marital_status"]))
    
    # Long loan terms - Only show if term is unusually long
    if user_inputs.get('loan_term', 36) > 60:  # More than 5 years
        relevant_sources.append(("Loan Term Impact", LOAN_KNOWLEDGE_BASE["loan_term"]))
    
    # Limit to top 6 most relevant sources to avoid overwhelming users
    return relevant_sources[:6]

# New function to display sources with better organization
def display_contextual_sources(relevant_sources, prediction_probability):
    """Display sources with better organization based on risk level"""
    
    risk_level = "High" if prediction_probability > 0.7 else "Moderate" if prediction_probability > 0.4 else "Low"
    
    st.write(f"**📊 Risk Level: {risk_level} ({prediction_probability*100:.1f}% chance of default)**")
    
    if prediction_probability > 0.5:
        st.write("**🚨 Key Risk Factors to Address:**")
    else:
        st.write("**✅ Factors Supporting This Application:**")
    
    for title, source_data in relevant_sources:
        with st.expander(f"📖 {title}"):
            st.write(source_data['content'].strip())
            
            # Display sources with links if available
            if source_data['sources']:
                st.write("**🔗 Learn More:**")
                for source in source_data['sources']:
                    st.markdown(f"- [{source['title']}]({source['url']})")

# Improved GPT-2 explanation generation with better error handling
def generate_explanation_with_gpt2(model, tokenizer, prediction_probability, user_inputs, relevant_sources):
    """Generate human-readable explanation using GPT-2 with improved error handling"""
    try:
        # Determine risk level and key factors
        risk_level = "high" if prediction_probability > 0.5 else "low"
        risk_percentage = prediction_probability * 100
        
        # Create a very focused and short prompt to avoid token issues
        prompt = f"This loan applicant has {risk_percentage:.0f}% default risk. "
        prompt += f"Income: ${user_inputs.get('income', 0):,}, "
        prompt += f"Credit Score: {user_inputs.get('credit_score', 0)}, "
        prompt += f"Loan: ${user_inputs.get('loan_amount', 0):,}. "
        prompt += "Simple explanation:"
        
        # Tokenize with strict limits
        inputs = tokenizer.encode(prompt, return_tensors='pt', max_length=200, truncation=True)
        
        # Check if inputs are valid
        if inputs.shape[1] == 0:
            raise ValueError("Input encoding failed")
        
        # Generate with conservative settings
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=min(inputs.shape[1] + 50, 300),  # Very conservative max length
                num_return_sequences=1,
                temperature=0.7,  # Lower temperature for more focused output
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        # Decode and clean response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        explanation = generated_text[len(prompt):].strip()
        
        # Clean up the explanation
        if explanation:
            # Remove incomplete sentences and clean up
            sentences = explanation.split('.')
            clean_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 15 and len(sentence) < 200:  # Reasonable sentence length
                    clean_sentences.append(sentence)
                if len(clean_sentences) >= 2:  # Limit to 2 sentences
                    break
            
            if clean_sentences:
                explanation = '. '.join(clean_sentences)
                if not explanation.endswith('.'):
                    explanation += '.'
            else:
                explanation = ""
        
        # Fallback if explanation is empty or too short
        if not explanation or len(explanation) < 20:
            return generate_simple_explanation(prediction_probability, user_inputs)
        
        return explanation
        
    except Exception as e:
        print(f"GPT-2 generation error: {str(e)}")  # For debugging
        # Always fallback to simple explanation
        return generate_simple_explanation(prediction_probability, user_inputs)

# Load scaler
try:
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Scaler file not found. Please ensure 'scaler.pkl' is in the same directory.")
    scaler = None

# Enhanced Streamlit Page Config and Custom Styling
#""st.set_page_config(
    #page_title="Loan Default Prediction Tool",
    #page_icon="💰",
    #layout="wide",
    #initial_sidebar_state="expanded"
#)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .creator-info {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #48dbfb 0%, #0abde3 100%);
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class="main-header">
    <h1>💰 Loan Default Prediction Tool</h1>
    <p>AI-Powered Risk Assessment for Smart Lending Decisions</p>
</div>
""", unsafe_allow_html=True)

# Creator Information
st.markdown("""
<div class="creator-info">
    <h3>🚀 Created by Raghav Sharma</h3>
    <p>Data Scientist & AI Engineer | Building Intelligent Financial Solutions</p>
    <p>
        <a href="https://raghav-sharma.com/" target="_blank" style="color: #ffd700; text-decoration: none;">
            🌐 Visit My Portfolio
        </a> | 
        <a href="https://github.com/sharmaraghav644/LoanDefaultPredictionApp" target="_blank" style="color: #ffd700; text-decoration: none;">
            📁 View Source Code
        </a>
        <a href="https://www.linkedin.com/in/raghav-sharma-b7a87a142/" target="_blank" style="color: #ffd700; text-decoration: none;">
            🔗 Visit My LinkedIn
    </p>
</div>
""", unsafe_allow_html=True)

# App Description with enhanced styling
st.markdown("### 🎯 About This Application")
st.markdown("""
This advanced loan default prediction tool uses machine learning algorithms to assess the risk of loan defaults. 
Built with **Streamlit**, powered by **XGBoost** and **Random Forest** models, 
and enhanced with AI-powered explanations to provide clear, actionable insights for lending decisions.
""")

st.markdown("#### 📊 Features:")
st.markdown("""
- 🤖 **AI-Powered Risk Assessment** - Advanced ML models for accurate predictions
- 📚 **Contextual Knowledge Base** - Relevant information based on your specific situation  
- 📈 **Business Insights** - Strategic recommendations for financial institutions
- 🎨 **Interactive Interface** - User-friendly design with real-time predictions
""")

st.markdown("""
**Dataset:** Built using the comprehensive [Loan Default Prediction Dataset from Kaggle](https://www.kaggle.com/datasets/nikhil1e9/loan-default/data)
""")

# Enhanced Sidebar with better styling
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem; color: white;">
    <h2>📝 Loan Application Form</h2>
    <p>Enter applicant details below</p>
</div>
""", unsafe_allow_html=True)
with st.sidebar.form("user_inputs"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Income", min_value=1000, max_value=1000000, value=50000)
    loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=500000, value=10000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    interest_rate = st.number_input("Interest Rate", min_value=2.0, max_value=25.0, value=10.0, step=0.1)
    months_employed = st.number_input("Months Employed", min_value=0, max_value=600, value=60)
    dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    loan_term = st.number_input("Loan Term (months)", min_value=12, max_value=360, value=36)

    education_mapping = {"Bachelor's": 0, "High School": 1, "Master's": 2, "PhD": 3}
    marital_status_mapping = {"Divorced": 0, "Married": 1, "Single": 2}
    employment_type_mapping = {"Full-time": 0, "Part-time": 1, "Self-employed": 2, "Unemployed": 3}
    has_co_signer_mapping = {"Yes": 1, "No": 0}
    has_mortgage_mapping = {"Yes": 1, "No": 0}
    has_dependents_mapping = {"Yes": 1, "No": 0}
    loan_purpose_mapping = {"Auto": 0, "Business": 1, "Education": 2, "Home": 3, "Other": 4}

    education = st.selectbox("Education", options=list(education_mapping.keys()))
    marital_status = st.selectbox("Marital Status", options=list(marital_status_mapping.keys()))
    employment_type = st.selectbox("Employment Type", options=list(employment_type_mapping.keys()))
    has_co_signer = st.selectbox("Has Co-Signer", options=list(has_co_signer_mapping.keys()))
    has_mortgage = st.selectbox("Has Mortgage", options=list(has_mortgage_mapping.keys()))
    has_dependents = st.selectbox("Has Dependents", options=list(has_dependents_mapping.keys()))
    loan_purpose = st.selectbox("Loan Purpose", options=list(loan_purpose_mapping.keys()))

    model_choice = st.selectbox("Select Model for Prediction", ["XGBoost", "Random Forest"])
    
    submitted = st.form_submit_button("Submit")

# If the form is submitted
if submitted:
    input_data = pd.DataFrame({
        "Age": [age], "Income": [income], "LoanAmount": [loan_amount], "CreditScore": [credit_score],
        "MonthsEmployed": [months_employed], "NumCreditLines": [int(df["NumCreditLines"].mean())],
        "InterestRate": [interest_rate], "LoanTerm": [loan_term], "DTIRatio": [dti_ratio],
        "Education_encoded": [education_mapping[education]],
        "EmploymentType_encoded": [employment_type_mapping[employment_type]],
        "MaritalStatus_encoded": [marital_status_mapping[marital_status]],
        "HasMortgage_encoded": [has_mortgage_mapping[has_mortgage]],
        "HasDependents_encoded": [has_dependents_mapping[has_dependents]],
        "LoanPurpose_encoded": [loan_purpose_mapping[loan_purpose]],
        "HasCoSigner_encoded": [has_co_signer_mapping[has_co_signer]],
    })
    input_data = input_data[scaler.feature_names_in_]
    scaled_data = scaler.transform(input_data)

    st.subheader("Prediction Results")
    with st.spinner("Loading model and predicting..."):
        model = get_model(model_choice)
        if model:
            probability = model.predict_proba(scaled_data)[:, 1][0]
            st.write(f"**Chances of Default: {float(probability) * 100:.2f}%**")
            
            # Prepare user inputs for RAG
            user_inputs_dict = {
                'age': age,
                'income': income,
                'loan_amount': loan_amount,
                'credit_score': credit_score,
                'interest_rate': interest_rate,
                'months_employed': months_employed,
                'dti_ratio': dti_ratio,
                'loan_term': loan_term,
                'education': education,
                'marital_status': marital_status,
                'employment_type': employment_type,
                'has_co_signer': has_co_signer,
                'has_mortgage': has_mortgage,
                'has_dependents': has_dependents,
                'loan_purpose': loan_purpose
            }
            
            st.subheader("🔍 Simple Risk Explanation")
            explanation = generate_simple_explanation(probability, user_inputs_dict)
            st.info(explanation)
            
            # Updated Knowledge Sources Section with contextual filtering
            st.subheader("📚 Relevant Information for Your Situation")
            
            relevant_sources = retrieve_relevant_sources(user_inputs_dict, probability)
            
            # Show only relevant sources with better organization
            if relevant_sources:
                display_contextual_sources(relevant_sources, probability)
            
            # Risk level indicator
            if probability > 0.7:
                st.error("⚠️ **High Risk**: This application shows significant risk factors that make default likely.")
            elif probability > 0.4:
                st.warning("⚡ **Moderate Risk**: This application has some concerning factors that need attention.")
            else:
                st.success("✅ **Low Risk**: This application shows good indicators for successful loan repayment.")

            st.subheader("Advanced Business Insights")
            if income > 100000 and education in ["Master's", "PhD"]:
                st.write("💡 **Targeted Loan Bundles**: Consider offering premium loans with lower interest rates for highly qualified, affluent borrowers.")
            if loan_amount > 40000 and income < 40000:
                st.write("⚠️ **Dynamic Loan Amount Caps**: High loan amounts in low-income brackets increase default risk. Adjust loan caps accordingly.")
            if loan_amount < 5000:
                st.write("📊 **Reevaluate Small Loan Policies**: High default rates suggest a need for microfinance coaching or flexible repayment plans.")
            if has_co_signer == "Yes":
                st.write("🤝 **Incentivize Co-Signed Loans**: Co-signed loans reduce default risk. Offering discounts for such cases can be beneficial.")