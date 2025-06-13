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
from sentence_transformers import SentenceTransformer
import faiss

warnings.filterwarnings('ignore')

# --- LOAN KNOWLEDGE BASE ---
# This dictionary holds detailed explanations and external sources for various loan factors.
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
# --- END LOAN KNOWLEDGE BASE ---


# Set page configuration only once
st.set_page_config(
    page_title="Loan Default Prediction Tool",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling (only once)
st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF; /* Main app background */
    }
    .stButton > button {
        background-color: #2E86AB;
        color: white;
    }
    /* Styles for headers and info boxes */
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: black; /* Changed to black for visibility */
    }
    
    .creator-info {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: black; /* Changed to black for visibility */
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white; /* Keep white if gradient is dark enough */
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
        color: black; /* Ensure text is visible */
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
        color: black; /* Ensure text is visible */
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white; /* Keep white for sidebar content */
    }

    /* General markdown text color */
    .stMarkdown, .stText {
        color: black; /* Ensure all general text is black */
    }

    /* Ensure selectbox and other input labels are visible */
    label {
        color: black !important;
    }

</style>
""", unsafe_allow_html=True)

# Load dataset (used for mean of NumCreditLines, if applicable)
@st.cache_data
def load_data():
    file_url = "https://raw.githubusercontent.com/sharmaraghav644/LoanDefaultPredictionApp/refs/heads/main/Loan_default_predication_kaggle.csv"
    try:
        df_loaded = pd.read_csv(file_url)
        return df_loaded
    except Exception as e:
        st.error(f"Could not load dataset from URL: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

df = load_data()
num_credit_lines_mean = int(df["NumCreditLines"].mean()) if not df.empty else 5 # Default to 5 if data not loaded


class RAGSystem:
    def __init__(self, knowledge_base_dict):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_base_dict = knowledge_base_dict
        # Extract only the content strings for embedding
        self._content_list = [entry["content"] for entry in self.knowledge_base_dict.values()]
        # Store a list of the values (content and sources dicts) corresponding to the content_list order
        self._knowledge_values = list(self.knowledge_base_dict.values())
        self.build_vector_store()
    
    def build_vector_store(self):
        embeddings = self.encoder.encode(self._content_list)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.embeddings = embeddings
    
    def get_context(self, query, top_k=3):
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        
        relevant_context = []
        for i, idx in enumerate(indices[0]):
            if scores[0][i] > 0.5:  # Increased similarity threshold for better relevance
                # Return the full structured entry (content and sources)
                relevant_context.append(self._knowledge_values[idx])
        return relevant_context

@st.cache_resource
def load_gpt2_model():
    """Load GPT-2 model and tokenizer (cached for performance)"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

class GPTExplainer:
    def __init__(self):
        self.model, self.tokenizer = load_gpt2_model()
    
    def explain_prediction(self, user_data, prediction_result, context):
        # Extract only content for GPT-2 prompt
        context_content = [item['content'] for item in context]
        context_text = " ".join(context_content[:3]) if context_content else "Standard risk assessment applies." # Use up to 3 context items
        
        prompt = f"Loan Analysis Report:\nBorrower: Age {user_data.get('age', 'N/A')}, Income ${user_data.get('income', 'N/A'):,}\nLoan Amount: ${user_data.get('loan_amount', 'N/A'):,}\nRisk Assessment: {prediction_result}\nKey Factors: {context_text}\nExplanation:"
        
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            explanation = generated_text[len(prompt):].strip()
            
            # Simple cleanup for generated text
            explanation = explanation.split('.')[0] + '.' if '.' in explanation else explanation
            
            if len(explanation) < 10:
                return self.simple_explanation(user_data, prediction_result, context)
            
            return f"**AI Analysis:** {explanation}"
            
        except Exception as e:
            return self.simple_explanation(user_data, prediction_result, context)
    
    def simple_explanation(self, user_data, prediction_result, context):
        explanation = f"**Prediction: {prediction_result}**\n\n"
        explanation += "**Key Risk Factors:**\n"
        # Use content from the structured context
        for i, ctx_item in enumerate(context[:2], 1):
            explanation += f"• {ctx_item['content'].split('.')[0]}...\n" # Take first sentence for conciseness
        
        if user_data.get('income') and user_data.get('loan_amount'):
            income_val = user_data['income'] if user_data['income'] != 0 else 1 
            ratio = user_data['loan_amount'] / income_val
            if ratio > 5:
                explanation += f"• High loan-to-income ratio ({ratio:.1f}x) increases risk significantly.\n"
            elif ratio > 3:
                explanation += f"• Moderate loan-to-income ratio ({ratio:.1f}x) requires careful evaluation.\n"
        
        return explanation
    
    # Removed answer_question as per previous request

# Function to load non-DL models safely
def load_bz2_model(file_path):
    try:
        with bz2.BZ2File(file_path, "rb") as f:
            return joblib.load(f)
    except FileNotFoundError:
        st.error(f"Model file not found: {file_path}. Please ensure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model from {file_path}: {e}")
        return None

@st.cache_resource
def get_model(choice):
    if choice == "XGBoost":
        return load_bz2_model("xgb_model_compressed.pkl.bz2")
    elif choice == "Random Forest":
        return load_bz2_model("rf_model_compressed.pkl.bz2")
    return None

def retrieve_relevant_sources(user_inputs, prediction_probability, rag_system_instance):
    query_parts = [
        f"loan default risk for {user_inputs.get('age', 'N/A')} year old",
        f"income ${user_inputs.get('income', 'N/A'):,}",
        f"loan amount ${user_inputs.get('loan_amount', 'N/A'):,}",
        f"credit score {user_inputs.get('credit_score', 'N/A')}",
        f"DTI ratio {user_inputs.get('dti_ratio', 'N/A'):.2f}",
        f"employment stability for {user_inputs.get('months_employed', 'N/A')} months",
        f"education level {user_inputs.get('education', 'N/A')}",
        f"has co-signer: {user_inputs.get('has_co_signer', 'N/A')}",
        f"loan purpose: {user_inputs.get('loan_purpose', 'N/A')}"
    ]
    full_query = " ".join(query_parts)

    # rag_context will now be a list of dictionaries like {"content": "...", "sources": [...]}
    rag_context = rag_system_instance.get_context(full_query, top_k=5)

    final_relevant_sources = []
    
    # Define a mapping for user-friendly titles based on keywords in content
    context_keyword_to_title = {
        "credit score": "Credit Score Analysis",
        "income": "Income & Repayment Capacity",
        "loan amount": "Loan Amount Considerations",
        "debt-to-income ratio": "Debt-to-Income (DTI) Impact",
        "interest rate": "Interest Rate Effect",
        "employment": "Employment Stability",
        "loan term": "Loan Term Dynamics",
        "education": "Education & Financial Literacy",
        "marital status": "Marital Status Influence",
        "loan purpose": "Loan Purpose Assessment",
        "co-signer": "Co-Signer Benefits",
        "mortgage": "Existing Mortgage Implications",
        "dependents": "Dependents & Expenses"
    }

    is_high_risk = prediction_probability > 0.5

    for item_data in rag_context: # item_data is now a dict with 'content' and 'sources'
        content_lower = item_data['content'].lower()
        title = "General Risk Factor" # Default title

        # Find the best matching title from our predefined mapping
        for keyword, mapped_title in context_keyword_to_title.items():
            if keyword in content_lower:
                title = mapped_title
                break
        
        # Apply specific filtering or prioritization based on user inputs and prediction
        # This logic decides which retrieved items are most relevant to show
        if "employment" in content_lower and user_inputs.get('months_employed', 60) < 24 and is_high_risk:
            final_relevant_sources.append((title, item_data))
        elif "loan amounts exceeding" in content_lower and (user_inputs.get('loan_amount', 0) / max(user_inputs.get('income', 1), 1)) > 3 and is_high_risk:
            final_relevant_sources.append((title, item_data))
        elif "interest rates" in content_lower and user_inputs.get('interest_rate', 0) > 10 and is_high_risk:
            final_relevant_sources.append((title, item_data))
        elif "education" in content_lower and user_inputs.get('education') == "High School" and is_high_risk:
            final_relevant_sources.append((title, item_data))
        elif "age" in content_lower and 25 <= user_inputs.get('age', 0) <= 35 and is_high_risk:
            final_relevant_sources.append((title, item_data))
        elif "bankruptcy" in content_lower: # If 'previous bankruptcy' was in KB, it would be caught here
            final_relevant_sources.append((title, item_data))
        else: # Always add relevant context that passed the initial RAG similarity threshold
            final_relevant_sources.append((title, item_data))

    unique_sources = []
    seen_contents = set()
    for title, source_data in final_relevant_sources:
        if source_data['content'] not in seen_contents:
            unique_sources.append((title, source_data))
            seen_contents.add(source_data['content'])

    return unique_sources[:6] # Limit to top 6 relevant sources for display

def display_contextual_sources(relevant_sources, prediction_probability):
    risk_level = "High" if prediction_probability > 0.7 else "Moderate" if prediction_probability > 0.4 else "Low"
    
    st.write(f"**📊 Risk Level: {risk_level} ({prediction_probability*100:.1f}% chance of default)**")
    
    if prediction_probability > 0.5:
        st.write("**🚨 Key Risk Factors to Address:**")
    else:
        st.write("**✅ Factors Supporting This Application:**")
    
    for title, source_data in relevant_sources:
        with st.expander(f"📖 {title}"):
            st.write(source_data['content'].strip())
            
            if source_data['sources']:
                st.write("**🔗 Learn More:**")
                for source in source_data['sources']:
                    st.markdown(f"- [{source['title']}]({source['url']})")

def generate_simple_explanation(prediction_probability, user_inputs):
    explanation = f"Based on the input data, the predicted chance of default is **{prediction_probability*100:.2f}%**.\n\n"
    if prediction_probability > 0.7:
        explanation += "This is considered a **High Risk** loan. Key concerns may include a high loan amount relative to income, lower credit score, or unstable employment.\n"
    elif prediction_probability > 0.4:
        explanation += "This is considered a **Moderate Risk** loan. There are some factors that could lead to default, and caution is advised. Review DTI ratio, interest rate, and employment stability.\n"
    else:
        explanation += "This is considered a **Low Risk** loan. The applicant demonstrates strong indicators for repayment, such as a good credit score and stable income.\n"

    if user_inputs.get('credit_score') is not None and user_inputs['credit_score'] < 650:
        explanation += f"- The credit score of {user_inputs['credit_score']} is below average, contributing to the risk.\n"
    if user_inputs.get('dti_ratio') is not None and user_inputs['dti_ratio'] > 0.4:
        explanation += f"- The Debt-to-Income ratio of {user_inputs['dti_ratio']:.2f} is higher than ideal, indicating potential financial strain.\n"
    if user_inputs.get('months_employed') is not None and user_inputs['months_employed'] < 12:
        explanation += f"- Employment length of {user_inputs['months_employed']} months is relatively short, which can increase perceived risk.\n"
    
    return explanation

def generate_explanation_with_gpt2(model, tokenizer, prediction_probability, user_inputs, relevant_sources):
    try:
        risk_level = "high" if prediction_probability > 0.5 else "low"
        risk_percentage = prediction_probability * 100
        
        # Extract content from relevant_sources for GPT-2 prompt
        context_contents = [src_item['content'] for _, src_item in relevant_sources if src_item['content']]
        context_str = " ".join(context_contents[:3]) # Use up to 3 relevant content sections

        prompt = f"This loan applicant has {risk_percentage:.0f}% default risk. "
        prompt += f"Their income is ${user_inputs.get('income', 0):,}, credit score is {user_inputs.get('credit_score', 0)}, and the requested loan amount is ${user_inputs.get('loan_amount', 0):,}. "
        if context_str:
            prompt += f"Consider the following: {context_str}. "
        prompt += "Provide a concise explanation of the risk assessment for a loan officer:"
        
        inputs = tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
        
        if inputs.shape[1] == 0:
            raise ValueError("Input encoding failed: Prompt too long or empty.")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=min(inputs.shape[1] + 80, 350),
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        explanation = generated_text[len(prompt):].strip()
        
        if explanation:
            sentences = explanation.split('.')
            clean_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10 and len(sentence) < 250:
                    clean_sentences.append(sentence)
                if len(clean_sentences) >= 2: # Aim for at least 2 good sentences
                    break
            
            if clean_sentences:
                explanation = '. '.join(clean_sentences)
                if not explanation.endswith('.') and explanation:
                    explanation += '.'
            else:
                explanation = ""
        
        if not explanation or len(explanation) < 30:
            return generate_simple_explanation(prediction_probability, user_inputs)
        
        return explanation
        
    except Exception as e:
        print(f"GPT-2 generation error: {str(e)}")
        return generate_simple_explanation(prediction_probability, user_inputs)

try:
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Scaler file 'scaler.pkl' not found. Please ensure it's in the same directory.")
    scaler = None
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    scaler = None


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
        </a> |
        <a href="https://www.linkedin.com/in/raghav-sharma-b7a87a142/" target="_blank" style="color: #ffd700; text-decoration: none;">
            🔗 Visit My LinkedIn
        </a>
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

# Initialize RAG system and GPT explainer
if 'rag_system' not in st.session_state:
    with st.spinner("Loading AI systems..."):
        # Pass the new LOAN_KNOWLEDGE_BASE to the RAGSystem
        st.session_state.rag_system = RAGSystem(LOAN_KNOWLEDGE_BASE) 
        st.session_state.gpt_explainer = GPTExplainer()

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
    if scaler is None:
        st.error("Cannot make prediction: Scaler not loaded. Please ensure 'scaler.pkl' is in the same directory and is valid.")
    else:
        input_data = pd.DataFrame({
            "Age": [age], "Income": [income], "LoanAmount": [loan_amount], "CreditScore": [credit_score],
            "MonthsEmployed": [months_employed], "NumCreditLines": [num_credit_lines_mean],
            "InterestRate": [interest_rate], "LoanTerm": [loan_term], "DTIRatio": [dti_ratio],
            "Education_encoded": [education_mapping[education]],
            "EmploymentType_encoded": [employment_type_mapping[employment_type]],
            "MaritalStatus_encoded": [marital_status_mapping[marital_status]],
            "HasMortgage_encoded": [has_mortgage_mapping[has_mortgage]],
            "HasDependents_encoded": [has_dependents_mapping[has_dependents]],
            "LoanPurpose_encoded": [loan_purpose_mapping[loan_purpose]],
            "HasCoSigner_encoded": [has_co_signer_mapping[has_co_signer]],
        })
        
        try:
            input_data = input_data[scaler.feature_names_in_]
            scaled_data = scaler.transform(input_data)
        except KeyError as e:
            st.error(f"Feature mismatch with scaler: {e}. Ensure your input features match the features the scaler was trained on. Expected features: {scaler.feature_names_in_}")
            st.stop()
        except Exception as e:
            st.error(f"Error during data scaling: {e}")
            st.stop()

        st.subheader("Prediction Results")
        with st.spinner("Loading model and predicting..."):
            model = get_model(model_choice)
            if model:
                probability = model.predict_proba(scaled_data)[:, 1][0]
                st.write(f"**Chances of Default: {float(probability) * 100:.2f}%**")
                
                user_inputs_dict = {
                    'age': age, 'income': income, 'loan_amount': loan_amount, 'credit_score': credit_score,
                    'interest_rate': interest_rate, 'months_employed': months_employed, 'dti_ratio': dti_ratio,
                    'loan_term': loan_term, 'education': education, 'marital_status': marital_status,
                    'employment_type': employment_type, 'has_co_signer': has_co_signer,
                    'has_mortgage': has_mortgage, 'has_dependents': has_dependents,
                    'loan_purpose': loan_purpose
                }
                
                st.subheader("🤖 AI-Powered Risk Analysis")

                query = f"loan default risk assessment for {user_inputs_dict.get('age', 30)} year old with ${user_inputs_dict.get('income', 50000)} income applying for ${user_inputs_dict.get('loan_amount', 25000)} loan. Also considering credit score {user_inputs_dict.get('credit_score', 0)} and DTI ratio {user_inputs_dict.get('dti_ratio', 0.0)}."
                
                # Get structured context (content and sources) from the RAG system
                context_for_explanation = st.session_state.rag_system.get_context(query)
                
                prediction_text = "High Risk" if probability > 0.5 else "Low Risk"

                # Pass the structured context to the explanation generator
                explanation = generate_explanation_with_gpt2(
                    st.session_state.gpt_explainer.model, 
                    st.session_state.gpt_explainer.tokenizer, 
                    probability, 
                    user_inputs_dict, 
                    context_for_explanation # Pass the full structured context
                )
                
                st.markdown(explanation)

                st.subheader("📚 Relevant Information for Your Situation")
                
                # The retrieve_relevant_sources function also gets the structured context
                relevant_sources_for_display = retrieve_relevant_sources(
                    user_inputs_dict, 
                    probability, 
                    st.session_state.rag_system
                )
                
                if relevant_sources_for_display:
                    display_contextual_sources(relevant_sources_for_display, probability)
                else:
                    st.info("No specific knowledge base context was highly relevant to these inputs for a deeper dive. General risk factors apply.")
                
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
            else:
                st.error("Model could not be loaded for prediction.")