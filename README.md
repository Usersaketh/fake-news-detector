# Fake News Verification System

A comprehensive fake news detection system built with Streamlit and Python, featuring multi-stage verification pipeline with AI models and fact-checking APIs.

## Features

### Input Types
- **Text Input**: Paste headline and article body directly
- **URL Input**: Fetch and extract content from web pages
- **Image Input**: AI-powered text extraction from screenshots, memes, or social media posts using Qwen vision model

### Verification Pipeline
1. **HuggingFace Quick Check**: Preliminary classification (separate panel)
2. **SERPER Search**: Multi-query evidence gathering from trusted sources
3. **FactCheck/NewsCheck APIs**: Official fact-checking database lookup
4. **DeepSeek R1 (Primary)**: Deep reasoning with evidence-based verdict
5. **Gemini (Fallback)**: Backup model for validation
6. **Aggregator**: Combines all claim verdicts into final conclusion

## Installation

### Prerequisites
- Python 3.8+

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd fake-news-detector
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
- **OPENROUTER_API_KEY**: Get from https://openrouter.ai/
- **SERPER_API_KEY**: Get from https://serper.dev/
- **GEMINI_API_KEY**: Get from https://makersuite.google.com/app/apikey
- **NEWSCHECK_API_KEY**: (Optional) NewsCheck API
- **FACTCHECK_API_KEY**: (Optional) FactCheck API

## Usage

### Run the application:
```bash
streamlit run app.py
```

### Testing with Examples

The system works with three input methods:

#### 1. Text Input
Paste headline and article body directly into the form.

#### 2. URL Input
Enter a news article URL like:
- https://www.bbc.com/news/...
- https://www.nytimes.com/...
- https://www.theguardian.com/...

#### 3. Image Input
Upload images containing text:
- Screenshots of social media posts
- Newspaper photos
- Memes with text overlays
- WhatsApp forwards

## Architecture

```
app.py                          # Streamlit UI with 4-section layout
backend/
├── __init__.py
├── utils.py                    # Caching, retry logic, helpers
├── extractor_text.py           # URL fetch, HTML parse, AI vision extraction
├── claim_extractor.py          # Extract verifiable claims
├── quick_hf.py                 # HuggingFace preliminary check
├── search_serper.py            # SERPER evidence search
├── factcheck_integrations.py   # NewsCheck + FactCheck APIs
├── deepseek_r1.py              # Primary reasoning model
├── gemini_fallback.py          # Fallback validation model
└── aggregator.py               # Combine verdicts
```

## UI Layout

### 1. Input Section
Three tabs for different input methods with preview

### 2. Quick Check (Left Column)
HuggingFace preliminary classification - NOT authoritative

### 3. Final Verdict (Center Column)
- Large colored verdict label
- Confidence score
- Primary source
- Expandable DeepSeek reasoning

### 4. Evidence Explorer (Right Column)
- SERPER search results with trust scores
- FactCheck API results
- Citation indicators

### 5. Raw Text (Collapsed)
Extracted text for transparency

## Models Used

### DeepSeek R1 (Primary)
- Model: `deepseek/deepseek-r1:free`
- Via: OpenRouter API
- Purpose: Evidence-based reasoning and verdict

### Gemini (Fallback)
- Model: `gemini-1.5-flash`
- Via: Google AI API
- Purpose: Backup when DeepSeek fails or returns low confidence

### Qwen Vision (Image Extraction)
- Model: `qwen/qwen2.5-vl-32b-instruct:free`
- Via: OpenRouter API
- Purpose: Intelligent text extraction from images, screenshots, and memes

### HuggingFace (Quick Check)
- Model: `Pulk17/Fake-News-Detection`
- Purpose: Fast preliminary classification

## Trust Scoring

Evidence sources are scored 0-1 based on:
- **+0.6**: Known fact-check sites (Snopes, FactCheck.org)
- **+0.4**: Reputable news outlets
- **+0.1**: Recent publication (< 30 days)
- **-0.2**: Unknown blogs or low-trust domains

## Verdict Logic

### Per-Claim Verdicts
- **TRUE**: Evidence strongly supports
- **FALSE**: Evidence contradicts
- **MIXED**: Conflicting evidence
- **INSUFFICIENT**: Not enough reliable evidence

### Overall Article Verdict
- Any claim FALSE (confidence > 0.85) → Article FALSE
- Majority TRUE → Article TRUE
- Multiple MIXED → Article MIXED
- All low confidence → INSUFFICIENT

## Error Handling

- Retry logic with exponential backoff
- Schema validation for all AI outputs
- Automatic fallback to Gemini
- Graceful degradation if APIs unavailable

## Caching

Results are cached to avoid redundant API calls:
- Evidence searches (24 hours)
- Fact-check API results (7 days)
- Model responses (1 hour)

## Troubleshooting

### Tesseract not found
Ensure Tesseract is installed and added to PATH.

### API Rate Limits
The system includes retry logic and respects rate limits.

### Invalid JSON from models
Automatic retry → fallback to Gemini → manual review flag

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License

## Disclaimer

This system is a tool to assist in fact-checking. Always verify important information through multiple trusted sources. AI models can make mistakes.
