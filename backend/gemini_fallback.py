"""
Gemini fallback model.
Used when DeepSeek fails, returns low confidence, or returns insufficient verdict.
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
    gemini_limiter
)

logger = logging.getLogger(__name__)


class GeminiFallback:
    """Gemini model as fallback verification."""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
    
    @cache_result(ttl_seconds=3600)
    def evaluate_claim(
        self,
        claim: Dict,
        evidence: List[Dict],
        api_results: List[Dict],
        deepseek_result: Dict = None
    ) -> Dict:
        """
        Evaluate claim using Gemini as fallback.
        
        Args:
            claim: Claim dict
            evidence: Evidence list
            api_results: API results
            deepseek_result: Original DeepSeek result (optional)
            
        Returns:
            Verdict dict
        """
        logger.info(f"Evaluating claim with Gemini fallback: {claim['claim_id']}")
        
        if not self.api_key:
            logger.warning("Gemini API key not found, returning error response")
            return self._error_response(claim['claim_id'])
        
        # Prepare evidence
        evidence_list = []
        
        for i, ev in enumerate(evidence[:10]):
            evidence_list.append({
                'id': f'e{i+1}',
                'url': ev['url'],
                'title': ev['title'],
                'snippet': ev['snippet'],
                'trust_score': ev['trust_score']
            })
        
        for i, api_ev in enumerate(api_results[:5]):
            evidence_list.append({
                'id': f'api{i+1}',
                'source': api_ev.get('source', 'api'),
                'url': api_ev.get('url', ''),
                'rating': api_ev.get('rating', api_ev.get('verdict', 'Unknown')),
                'trust_score': api_ev.get('trust_score', 0.9)
            })
        
        prompt = f"""You are a fallback fact-checking model. Evaluate the following claim using ONLY the structured evidence provided.

Claim: "{claim['text']}"

Evidence:
{json.dumps(evidence_list, indent=2)}

Return STRICT JSON in this format:
{{
  "verdict": "true|false|mixed|insufficient",
  "confidence": 0.0,
  "explanation": "brief explanation citing evidence"
}}

Rules:
- Base verdict ONLY on provided evidence
- If evidence is insufficient, say "insufficient"
- Cite evidence IDs in explanation
- Output ONLY valid JSON"""

        try:
            gemini_limiter.wait_if_needed()
            result = self._call_gemini_api(prompt)
            
            if self._validate_result(result):
                # Add claim_id
                result['claim_id'] = claim['claim_id']
                result['model'] = 'gemini_fallback'
                
                logger.info(f"Gemini evaluation successful: {result['verdict']}")
                return result
            else:
                logger.warning("Gemini returned invalid response")
                return self._error_response(claim['claim_id'])
                
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return self._error_response(claim['claim_id'])
    
    @retry_on_failure(max_attempts=2)
    def _call_gemini_api(self, prompt: str) -> Dict:
        """
        Call Gemini API.
        
        Args:
            prompt: Prompt text
            
        Returns:
            Parsed response dict
        """
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            # Use correct model name without models/ prefix
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=1000,
                )
            )
            
            result_text = response.text.strip()
            logger.info(f"Gemini raw response (first 300 chars): {result_text[:300]}")
            
            # Parse JSON
            result = parse_json_response(result_text)
            
            if not result:
                logger.error(f"Failed to parse Gemini response. Raw: {result_text[:500]}")
                raise ValueError("Failed to parse Gemini response as JSON")
            
            return result
            
        except ImportError:
            logger.error("google-generativeai package not installed")
            raise
    
    def _validate_result(self, result: Dict) -> bool:
        """
        Validate Gemini result format.
        
        Args:
            result: Result dict to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["verdict", "confidence", "explanation"]
        
        if not validate_json_schema(result, required_fields):
            return False
        
        if result['verdict'] not in ['true', 'false', 'mixed', 'insufficient']:
            return False
        
        try:
            confidence = float(result['confidence'])
            if not (0.0 <= confidence <= 1.0):
                return False
        except:
            return False
        
        return True
    
    def _error_response(self, claim_id: str) -> Dict:
        """Generate error response."""
        return {
            "claim_id": claim_id,
            "verdict": "insufficient",
            "confidence": 0.0,
            "explanation": "Error: Unable to evaluate claim (fallback also failed).",
            "model": "gemini_fallback",
            "error": True
        }


def should_use_fallback(deepseek_result: Dict) -> bool:
    """
    Determine if Gemini fallback should be used.
    
    Args:
        deepseek_result: Result from DeepSeek
        
    Returns:
        True if fallback should be used
    """
    # Use fallback if error flag is set
    if deepseek_result.get('error', False):
        return True
    
    # Use fallback if needs_fallback flag is set
    if deepseek_result.get('needs_fallback', False):
        return True
    
    # Use fallback if verdict is insufficient
    if deepseek_result.get('verdict') == 'insufficient':
        return True
    
    # Use fallback only if confidence is VERY low (< 0.3)
    # Changed from 0.5 to 0.3 to avoid unnecessary fallbacks
    if deepseek_result.get('confidence', 1.0) < 0.3:
        return True
    
    return False


def apply_gemini_fallback(
    claims: List[Dict],
    deepseek_verdicts: List[Dict],
    evidence_map: Dict[str, List[Dict]],
    api_results_map: Dict[str, List[Dict]]
) -> List[Dict]:
    """
    Apply Gemini fallback to claims that need it.
    
    Args:
        claims: List of claim dicts
        deepseek_verdicts: List of DeepSeek verdicts
        evidence_map: Evidence mapping
        api_results_map: API results mapping
        
    Returns:
        List of final verdicts (DeepSeek or Gemini)
    """
    gemini = GeminiFallback()
    final_verdicts = []
    
    claim_map = {c['claim_id']: c for c in claims}
    
    for ds_verdict in deepseek_verdicts:
        claim_id = ds_verdict['claim_id']
        
        if should_use_fallback(ds_verdict):
            logger.info(f"Using Gemini fallback for {claim_id}")
            
            claim = claim_map[claim_id]
            evidence = evidence_map.get(claim_id, [])
            api_results = api_results_map.get(claim_id, [])
            
            gemini_verdict = gemini.evaluate_claim(
                claim,
                evidence,
                api_results,
                ds_verdict
            )
            
            # If Gemini also fails (returns error), keep the original DeepSeek verdict
            if gemini_verdict.get('error', False):
                logger.warning(f"Gemini fallback failed for {claim_id}, keeping DeepSeek verdict")
                final_verdicts.append(ds_verdict)
            else:
                final_verdicts.append(gemini_verdict)
        else:
            # Keep DeepSeek result
            final_verdicts.append(ds_verdict)
    
    return final_verdicts
