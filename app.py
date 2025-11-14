"""
Fake News Verification System - Streamlit UI
Complete multi-stage verification pipeline with clean separated output.
"""

import os
import warnings
import streamlit as st
from dotenv import load_dotenv
import logging

# Suppress PyTorch warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import backend modules
from backend.extractor_text import extract_text
from backend.claim_extractor import extract_claims
from backend.quick_hf import perform_quick_check
from backend.search_serper import search_for_evidence
from backend.factcheck_integrations import check_claims_with_apis
from backend.deepseek_r1 import evaluate_claims_with_deepseek
from backend.gemini_fallback import apply_gemini_fallback
from backend.aggregator import aggregate_verdicts
from backend.utils import (
    get_verdict_color,
    get_verdict_emoji,
    format_confidence,
    truncate_text
)

# Page config
st.set_page_config(
    page_title="Fake News Verification System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
        color: white;
    }
    .verdict-card {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .verdict-label {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .confidence-score {
        font-size: 1.5rem;
        margin: 0.5rem 0;
    }
    .evidence-item {
        padding: 1rem;
        border-left: 3px solid #667eea;
        margin: 0.5rem 0;
        background: #f8f9fa;
        border-radius: 5px;
    }
    .trust-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .trust-high { background: #d4edda; color: #155724; }
    .trust-medium { background: #fff3cd; color: #856404; }
    .trust-low { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Fake News Verification System</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-weight: bold;'>Multi-stage AI-powered fact-checking pipeline</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize session history in session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 AI Models Used")
        
        with st.expander("**DeepSeek R1** - Primary Reasoning", expanded=False):
            st.markdown("""
            **Role:** Primary fact-checking engine
            
            **Capabilities:**
            - Deep reasoning and logical analysis
            - Context-aware verdict generation
            - Confidence scoring (0.0-1.0)
            - Evidence synthesis from multiple sources
            
            **Access:** Via OpenRouter API
            """)
        
        with st.expander("**Qwen 2.5 Vision** - Image OCR", expanded=False):
            st.markdown("""
            **Role:** Image text extraction
            
            **Capabilities:**
            - Multi-language OCR
            - Screenshot text extraction
            - Handwriting recognition
            - Layout-aware text parsing
            
            **Access:** Via OpenRouter API
            """)
        
        with st.expander("**Gemini 1.5 Flash** - Fallback", expanded=False):
            st.markdown("""
            **Role:** Backup verification system
            
            **Activation:** When DeepSeek confidence < 0.3
            
            **Capabilities:**
            - Fast secondary verification
            - Cross-validation of low-confidence claims
            - Alternative reasoning paths
            
            **Access:** Direct Google Generative AI API
            """)
        
        with st.expander("**HuggingFace Models** - Quick Check", expanded=False):
            st.markdown("""
            **Models (Ensemble):**
            1. Pulk17/Fake-News-Detection
            2. vikram71198/distilroberta-base
            3. jy46604790/Fake-News-Bert
            4. winterForestStump/Roberta-fake-news
            
            **Role:** Preliminary fast screening
            
            **Method:** Ensemble voting with confidence averaging
            
            **Note:** For reference only, not authoritative
            
            **Output:** Real/Fake classification with vote breakdown
            """)
        
        with st.expander("**SERPER API** - Evidence Search", expanded=False):
            st.markdown("""
            **Role:** Web evidence gathering
            
            **Capabilities:**
            - Google Search integration
            - Trust score calculation
            - Domain authority ranking
            - Snippet extraction
            """)
        
        with st.expander("**FactCheck APIs** - Databases", expanded=False):
            st.markdown("""
            **Services:** Google FactCheck, NewsCheck
            
            **Role:** Pre-verified fact database lookup
            
            **Coverage:**
            - Professional fact-checking organizations
            - Historical claims database
            - Curated authoritative sources
            """)
        
        st.markdown("---")
        
        # Session History
        st.header("📜 Recent History")
        if st.session_state.history:
            for idx, entry in enumerate(reversed(st.session_state.history[-5:]), 1):
                verdict = entry['verdict']
                confidence = entry['confidence']
                headline = entry['headline']
                timestamp = entry['timestamp']
                
                verdict_color = get_verdict_color(verdict)
                verdict_emoji = get_verdict_emoji(verdict)
                
                with st.expander(f"{verdict_emoji} {truncate_text(headline, 30)}", expanded=False):
                    st.markdown(f"**Verdict:** :{'green' if verdict == 'true' else 'red' if verdict == 'false' else 'orange'}[{verdict.upper()}]")
                    st.markdown(f"**Confidence:** {format_confidence(confidence)}")
                    st.caption(f"🕒 {timestamp}")
        else:
            st.info("No verification history yet")
        
        st.markdown("---")
        st.header("⚙️ Settings")
        
        verification_mode = st.radio(
            "Verification Mode:",
            ["Single Article Check", "Multi-Claim Analysis"],
            help="Single: Verify entire article as one claim. Multi: Extract and verify multiple claims separately."
        )
        
        if verification_mode == "Multi-Claim Analysis":
            max_claims = st.slider("Max claims to extract", 1, 10, 3)
        else:
            max_claims = 1  # Only headline for single check
            
        show_raw_text = st.checkbox("Show raw extracted text", value=False)
        show_evidence_details = st.checkbox("Show all evidence details", value=True)
    
    # ==================== SECTION 1: INPUT ====================
    st.header("📝 Step 1: Input Selection")
    
    input_tab1, input_tab2, input_tab3 = st.tabs(["📄 Text Input", "🔗 URL Input", "🖼️ Image Input"])
    
    extracted_data = None
    
    with input_tab1:
        st.subheader("Enter text directly")
        headline_input = st.text_input("Headline:", placeholder="Enter article headline...")
        body_input = st.text_area("Article Body:", height=200, placeholder="Paste article text here...")
        
        if st.button("🔍 Verify Text", key="btn_text"):
            if headline_input or body_input:
                with st.spinner("Extracting text..."):
                    extracted_data = extract_text('text', headline=headline_input, body=body_input)
                    st.session_state['extracted_data'] = extracted_data
            else:
                st.warning("Please enter headline or body text.")
    
    with input_tab2:
        st.subheader("Fetch from URL")
        url_input = st.text_input("Article URL:", placeholder="https://example.com/article")
        
        if st.button("🔍 Verify URL", key="btn_url"):
            if url_input:
                with st.spinner("Fetching and extracting from URL..."):
                    try:
                        extracted_data = extract_text('url', url=url_input)
                        st.session_state['extracted_data'] = extracted_data
                        st.success(f"✅ Successfully extracted from {url_input}")
                    except Exception as e:
                        st.error(f"❌ Failed to extract from URL: {str(e)}")
            else:
                st.warning("Please enter a URL.")
    
    with input_tab3:
        st.subheader("Upload image (OCR)")
        st.caption("*Powered by Qwen Vision Model via OpenRouter*")
        image_file = st.file_uploader("Upload image:", type=['png', 'jpg', 'jpeg', 'bmp'])
        
        if st.button("🔍 Verify Image", key="btn_image"):
            if image_file:
                with st.spinner("Extracting text from image using AI vision model..."):
                    try:
                        extracted_data = extract_text('image', image_file=image_file)
                        st.session_state['extracted_data'] = extracted_data
                        
                        # Check if extraction failed due to API key missing
                        error_type = extracted_data['metadata'].get('error')
                        if error_type == 'api_key_missing':
                            st.error("❌ OpenRouter API key not configured!")
                            st.info(extracted_data['body'])
                        elif error_type:
                            st.error(f"❌ Image extraction failed: {extracted_data['body']}")
                        else:
                            st.success("✅ Text extracted from image using AI vision model")
                    except Exception as e:
                        st.error(f"❌ Extraction failed: {str(e)}")
            else:
                st.warning("Please upload an image.")
    
    # Use session state if available
    if 'extracted_data' in st.session_state:
        extracted_data = st.session_state['extracted_data']
    
    # Show extracted text preview
    if extracted_data:
        st.markdown("---")
        st.subheader("📋 Extracted Content Preview")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Headline:**")
            st.info(extracted_data['headline'])
        
        with col2:
            st.markdown("**Body Preview:**")
            preview = truncate_text(extracted_data['body'], 200)
            st.info(preview)
        
        # ==================== PROCESSING ====================
        st.markdown("---")
        st.header("🔄 Step 2: Multi-Stage Verification")
        
        if st.button("🚀 Start Full Verification", type="primary", use_container_width=True):
            run_verification_pipeline(extracted_data, max_claims, show_raw_text, show_evidence_details)


def run_verification_pipeline(extracted_data, max_claims, show_raw_text, show_evidence_details):
    """Run the complete verification pipeline."""
    
    # Extract claims
    with st.spinner("⚙️ Stage 1/6: Extracting verifiable claims..."):
        claims = extract_claims(
            extracted_data['headline'],
            extracted_data['body'],
            max_claims=max_claims
        )
        st.success(f"✅ Extracted {len(claims)} claims")
    
    # Quick HF check
    with st.spinner("⚙️ Stage 2/6: Running HuggingFace ensemble check (4 models)..."):
        full_text = extracted_data['full_text'][:1000]
        hf_result = perform_quick_check(full_text)
        st.success("✅ Ensemble quick check complete")
    
    # Search for evidence
    with st.spinner("⚙️ Stage 3/6: Searching for evidence (SERPER)..."):
        evidence_map = search_for_evidence(claims)
        total_evidence = sum(len(ev) for ev in evidence_map.values())
        st.success(f"✅ Found {total_evidence} evidence items")
    
    # Check fact-check APIs
    with st.spinner("⚙️ Stage 4/6: Checking FactCheck/NewsCheck APIs..."):
        api_results_map = check_claims_with_apis(claims)
        total_api_results = sum(len(res) for res in api_results_map.values())
        st.success(f"✅ Found {total_api_results} API fact-checks")
    
    # DeepSeek evaluation
    with st.spinner("⚙️ Stage 5/6: Deep reasoning with DeepSeek R1..."):
        deepseek_verdicts = evaluate_claims_with_deepseek(
            claims,
            evidence_map,
            api_results_map
        )
        st.success("✅ DeepSeek evaluation complete")
    
    # Gemini fallback (if needed)
    with st.spinner("⚙️ Stage 6/6: Applying Gemini fallback (if needed)..."):
        final_verdicts = apply_gemini_fallback(
            claims,
            deepseek_verdicts,
            evidence_map,
            api_results_map
        )
        st.success("✅ Fallback check complete")
    
    # Aggregate results
    with st.spinner("📊 Aggregating final verdict..."):
        overall_result = aggregate_verdicts(final_verdicts, evidence_map)
    
    st.success("🎉 Verification complete!")
    st.markdown("---")
    
    # Save to session history
    from datetime import datetime
    st.session_state.history.append({
        'verdict': overall_result['overall_verdict'],
        'confidence': overall_result['overall_confidence'],
        'headline': extracted_data['headline'][:100],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # ==================== SECTION 2: RESULTS ====================
    display_results(hf_result, overall_result, evidence_map, show_raw_text, extracted_data, show_evidence_details)


def display_results(hf_result, overall_result, evidence_map, show_raw_text, extracted_data, show_evidence_details):
    """Display verification results in organized sections."""
    
    st.header("📊 Verification Results")
    
    # Create three columns for results
    col_left, col_center, col_right = st.columns([1, 1.5, 1.5])
    
    # ==================== LEFT: HF QUICK CHECK ====================
    with col_left:
        st.subheader("⚡ Multi-Model Quick Check")
        st.caption("*HuggingFace Models - Preliminary Ensemble*")
        
        # Ensemble result
        hf_label = hf_result.get('label', 'uncertain')
        hf_confidence = hf_result.get('confidence', 0.0)
        hf_verdict = hf_result.get('verdict', 'Unknown')
        hf_model = hf_result.get('model', 'HuggingFace')
        
        # Color based on label
        if 'true' in hf_label:
            hf_color = "#2ecc71"
        elif 'false' in hf_label:
            hf_color = "#e74c3c"
        else:
            hf_color = "#95a5a6"
        
        # Display individual model results
        if 'individual_results' in hf_result:
            st.markdown("**Individual Model Results:**")
            for result in hf_result['individual_results']:
                model_name = result['model']
                verdict = result['verdict']
                confidence = result['confidence']
                
                # Color for individual result
                if result['label'] == 'real':
                    color = "#2ecc71"
                    icon = "✅"
                else:
                    color = "#e74c3c"
                    icon = "❌"
                
                st.markdown(f"""
                <div style="padding: 0.5rem; background: {color}10; border-left: 3px solid {color}; border-radius: 3px; margin: 0.3rem 0;">
                    <div style="font-size: 0.8rem; font-weight: 600; color: {color};">
                        {icon} {model_name}
                    </div>
                    <div style="font-size: 0.75rem; color: #666;">
                        {verdict} ({format_confidence(confidence)})
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("**Summary:**")
        st.write(hf_result.get('summary', 'No summary available'))
        
        # Show note if using fallback
        if hf_result.get('note'):
            st.info(hf_result['note'])
        
        st.warning("⚠️ Not authoritative - For quick reference only")
    
    # ==================== CENTER: FINAL VERDICT ====================
    with col_center:
        st.subheader("🎯 Final Verdict")
        st.caption("*DeepSeek R1 → Gemini Fallback*")
        
        verdict = overall_result['overall_verdict']
        confidence = overall_result['overall_confidence']
        primary_source = overall_result['primary_source']
        
        verdict_color = get_verdict_color(verdict)
        verdict_emoji = get_verdict_emoji(verdict)
        
        # Large verdict card
        st.markdown(f"""
        <div class="verdict-card" style="background: {verdict_color}20; border: 3px solid {verdict_color};">
            <div class="verdict-label" style="color: {verdict_color};">
                {verdict_emoji} {verdict.upper()}
            </div>
            <div class="confidence-score">
                Confidence: {format_confidence(confidence)}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Primary source
        st.markdown("**Primary Source:**")
        if primary_source['url']:
            st.markdown(f"[{primary_source['title']}]({primary_source['url']})")
            st.caption(f"Trust score: {format_confidence(primary_source['trust_score'])}")
        else:
            st.info("No authoritative source found")
        
        # Verdict breakdown
        with st.expander("📈 Verdict Breakdown"):
            breakdown = overall_result['verdict_breakdown']
            st.write(f"- ✅ TRUE: {breakdown['true']}")
            st.write(f"- ❌ FALSE: {breakdown['false']}")
            st.write(f"- ⚠️ MIXED: {breakdown['mixed']}")
            st.write(f"- ❓ INSUFFICIENT: {breakdown['insufficient']}")
        
        # Show detailed explanations
        with st.expander("📝 Show Detailed Explanations"):
            for claim_verdict in overall_result['claims']:
                st.markdown(f"**Claim:** {claim_verdict.get('claim_id', 'Unknown')}")
                st.write(f"*Verdict:* {claim_verdict.get('verdict', 'Unknown').upper()}")
                st.write(f"*Confidence:* {format_confidence(claim_verdict.get('confidence', 0.0))}")
                st.write(f"*Explanation:* {claim_verdict.get('explanation', 'No explanation')}")
                
                if claim_verdict.get('model') == 'gemini_fallback':
                    st.caption("🔄 *Gemini fallback used*")
                
                st.markdown("---")
    
    # ==================== RIGHT: EVIDENCE EXPLORER ====================
    with col_right:
        st.subheader("🔎 Evidence Explorer")
        st.caption("*SERPER + FactCheck APIs*")
        
        # Combine all evidence
        all_evidence = []
        for claim_id, evidence_list in evidence_map.items():
            all_evidence.extend(evidence_list)
        
        # Sort by trust score
        all_evidence.sort(key=lambda x: x.get('trust_score', 0.0), reverse=True)
        
        # Display top evidence
        num_to_show = len(all_evidence) if show_evidence_details else min(5, len(all_evidence))
        
        for i, ev in enumerate(all_evidence[:num_to_show]):
            trust_score = ev.get('trust_score', 0.0)
            
            # Trust badge
            if trust_score >= 0.7:
                badge_class = "trust-high"
                badge_text = "HIGH TRUST"
            elif trust_score >= 0.4:
                badge_class = "trust-medium"
                badge_text = "MEDIUM TRUST"
            else:
                badge_class = "trust-low"
                badge_text = "LOW TRUST"
            
            st.markdown(f"""
            <div class="evidence-item">
                <div>
                    <span class="trust-badge {badge_class}">{badge_text}</span>
                    <span style="margin-left: 0.5rem; color: #666;">
                        {format_confidence(trust_score)}
                    </span>
                </div>
                <div style="margin-top: 0.5rem;">
                    <a href="{ev.get('url', '#')}" target="_blank" style="font-weight: bold; color: #667eea;">
                        {ev.get('title', 'Untitled')}
                    </a>
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #555;">
                    {truncate_text(ev.get('snippet', ''), 150)}
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #888;">
                    {ev.get('domain', 'Unknown domain')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if len(all_evidence) > num_to_show:
            st.info(f"Showing {num_to_show} of {len(all_evidence)} evidence items")
    
    # ==================== BOTTOM: RAW TEXT ====================
    if show_raw_text:
        st.markdown("---")
        with st.expander("📄 Raw Extracted Text", expanded=False):
            st.text_area("Full Text:", extracted_data['full_text'], height=300)


if __name__ == "__main__":
    main()
