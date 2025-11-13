"""
DeepSeek R1 integration via OpenRouter.
Primary reasoning model for evidence-based verdict.
"""

import os
import json
from typing import Dict, List
import logging
from backend.utils import (
    cache_result,
    retry_on_failure,
    parse_json_response,
    validate_json_schema,
    openrouter_limiter
)

logger = logging.getLogger(__name__)


class DeepSeekR1:
    """DeepSeek R1 model for deep reasoning and verdict."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = "deepseek/deepseek-r1-0528-qwen3-8b:free"
        self.referer = os.getenv("HTTP_REFERER", "https://fake-news-detector.com")
        self.title = os.getenv("HTTP_TITLE", "Fake News Detector")
    
    @cache_result(ttl_seconds=3600)
    def evaluate_claim(
        self,
        claim: Dict,
        evidence: List[Dict],
        api_results: List[Dict]
    ) -> Dict:
        """
        Evaluate a single claim using evidence.
        
        Args:
            claim: Claim dict with claim_id and text
            evidence: List of evidence dicts from SERPER
            api_results: List of results from FactCheck/NewsCheck APIs
            
        Returns:
            Verdict dict with verdict, confidence, explanation, cited_evidence
        """
        logger.info(f"Evaluating claim with DeepSeek R1: {claim['claim_id']}")
        
        # Prepare evidence list with IDs
        evidence_list = []
        
        # Add SERPER evidence
        for i, ev in enumerate(evidence[:10]):  # Limit to top 10
            evidence_list.append({
                'id': f'e{i+1}',
                'source': 'web_search',
                'url': ev['url'],
                'title': ev['title'],
                'snippet': ev['snippet'],
                'trust_score': ev['trust_score']
            })
        
        # Add API evidence (high priority)
        for i, api_ev in enumerate(api_results[:5]):
            evidence_list.append({
                'id': f'api{i+1}',
                'source': api_ev.get('source', 'fact_check_api'),
                'url': api_ev.get('url', ''),
                'title': api_ev.get('title', ''),
                'rating': api_ev.get('rating', api_ev.get('verdict', 'Unknown')),
                'trust_score': api_ev.get('trust_score', 0.9)
            })
        
        # Build prompt
        system_message = """You are a fact-checking AI. You MUST respond with ONLY valid JSON, nothing else.

CRITICAL: Do not include any reasoning, thinking, or explanation outside the JSON structure.

Task: Evaluate the claim using ONLY the evidence provided.

Verdict options:
- "true": Evidence strongly supports
- "false": Evidence contradicts  
- "mixed": Evidence conflicts
- "insufficient": Not enough evidence

Output format: Pure JSON only, no markdown, no text before or after."""

        user_message = f"""Claim: "{claim['text']}"

Evidence:
{json.dumps(evidence_list, indent=2)}

Respond with ONLY this JSON structure (no other text):
{{
  "claim_id": "{claim['claim_id']}",
  "verdict": "true|false|mixed|insufficient",
  "confidence": 0.85,
  "explanation": "Brief explanation citing [e1] [e2] etc",
  "cited_evidence": [{{"id": "e1", "url": "...", "quote": "..."}}]
}}"""

        try:
            openrouter_limiter.wait_if_needed()
            result = self._call_deepseek_api(system_message, user_message)
            
            # Validate result
            if self._validate_result(result):
                logger.info(f"DeepSeek evaluation successful: {result['verdict']}")
                return result
            else:
                logger.warning("DeepSeek returned invalid response, retrying once")
                # Retry once
                openrouter_limiter.wait_if_needed()
                result = self._call_deepseek_api(system_message, user_message)
                
                if self._validate_result(result):
                    return result
                else:
                    logger.error("DeepSeek retry also failed")
                    return self._error_response(claim['claim_id'])
                    
        except Exception as e:
            logger.error(f"DeepSeek API call failed: {e}")
            return self._error_response(claim['claim_id'])
    
    @retry_on_failure(max_attempts=2)
    def _call_deepseek_api(self, system_message: str, user_message: str) -> Dict:
        """
        Call DeepSeek via OpenRouter API.
        
        Args:
            system_message: System prompt
            user_message: User prompt
            
        Returns:
            Parsed response dict
        """
        from openai import OpenAI
        import time as time_module
        
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            timeout=60.0,  # Increase timeout for longer responses
            max_retries=3  # Let OpenAI client handle retries for 429 errors
        )
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,  # Lower temperature for more consistent output
            max_tokens=4000,  # Increased to prevent truncation
            extra_headers={
                "HTTP-Referer": self.referer,
                "X-Title": self.title
            }
        )
        
        result_text = response.choices[0].message.content
        
        # Check for empty response
        if not result_text or not result_text.strip():
            logger.error("DeepSeek returned empty response")
            raise ValueError("Empty response from DeepSeek")
        
        result_text = result_text.strip()
        logger.info(f"DeepSeek raw response (first 500 chars): {result_text[:500]}")
        logger.info(f"DeepSeek response length: {len(result_text)} chars")
        
        # DeepSeek R1 models often output reasoning before JSON
        # Try multiple parsing strategies
        result = parse_json_response(result_text)
        
        if not result:
            # Try to find JSON after reasoning markers
            for marker in ["```json", "Answer:", "Output:", "Result:", "JSON:", "```"]:
                if marker in result_text:
                    try:
                        start_idx = result_text.index(marker) + len(marker)
                        json_part = result_text[start_idx:].strip()
                        
                        # Remove closing ``` if present
                        if json_part.endswith("```"):
                            json_part = json_part[:-3].strip()
                        
                        # Try to fix incomplete JSON by adding closing braces
                        if json_part and not json_part.endswith("}"):
                            # Count opening and closing braces
                            open_count = json_part.count("{")
                            close_count = json_part.count("}")
                            if open_count > close_count:
                                missing = open_count - close_count
                                json_part += "}" * missing
                                logger.info(f"Added {missing} closing braces to fix incomplete JSON")
                        
                        result = parse_json_response(json_part)
                        if result:
                            logger.info(f"Successfully parsed JSON after marker: {marker}")
                            break
                    except Exception as e:
                        logger.debug(f"Failed to parse after marker {marker}: {e}")
                        continue
        
        if not result:
            logger.error(f"All JSON parsing attempts failed. Raw response: {result_text[:1000]}")
            raise ValueError("Failed to parse DeepSeek response as JSON")
        
        return result
    
    def _validate_result(self, result: Dict) -> bool:
        """
        Validate DeepSeek result format.
        
        Args:
            result: Result dict to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["claim_id", "verdict", "confidence", "explanation"]
        
        if not validate_json_schema(result, required_fields):
            return False
        
        # Validate verdict value
        if result['verdict'] not in ['true', 'false', 'mixed', 'insufficient']:
            return False
        
        # Validate confidence is number
        try:
            confidence = float(result['confidence'])
            if not (0.0 <= confidence <= 1.0):
                return False
        except:
            return False
        
        return True
    
    def _error_response(self, claim_id: str) -> Dict:
        """
        Generate error response when API fails.
        
        Args:
            claim_id: ID of the claim
            
        Returns:
            Error response dict
        """
        return {
            "claim_id": claim_id,
            "verdict": "insufficient",
            "confidence": 0.0,
            "explanation": "Error: Unable to evaluate claim due to API failure.",
            "cited_evidence": [],
            "error": True,
            "needs_fallback": True
        }


def evaluate_claims_with_deepseek(
    claims: List[Dict],
    evidence_map: Dict[str, List[Dict]],
    api_results_map: Dict[str, List[Dict]]
) -> List[Dict]:
    """
    Evaluate multiple claims using DeepSeek R1.
    
    Args:
        claims: List of claim dicts
        evidence_map: Dict mapping claim_id to evidence list
        api_results_map: Dict mapping claim_id to API results
        
    Returns:
        List of verdict dicts
    """
    deepseek = DeepSeekR1()
    verdicts = []
    
    for claim in claims:
        claim_id = claim['claim_id']
        evidence = evidence_map.get(claim_id, [])
        api_results = api_results_map.get(claim_id, [])
        
        verdict = deepseek.evaluate_claim(claim, evidence, api_results)
        verdicts.append(verdict)
    
    return verdicts
