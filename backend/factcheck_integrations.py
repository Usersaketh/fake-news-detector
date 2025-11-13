"""
FactCheck and NewsCheck API integrations.
Queries official fact-checking databases for verified claims.
"""

import os
import json
import requests
from typing import List, Dict, Optional
import logging
from backend.utils import cache_result, retry_on_failure

logger = logging.getLogger(__name__)


class FactCheckAPI:
    """Interface to FactCheck API."""
    
    def __init__(self):
        self.api_key = os.getenv("FACTCHECK_API_KEY")
        self.api_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    
    @cache_result(ttl_seconds=604800)  # Cache for 7 days
    def check_claim(self, claim: str) -> List[Dict]:
        """
        Check claim against FactCheck database.
        
        Args:
            claim: Claim text to check
            
        Returns:
            List of fact check results
        """
        if not self.api_key:
            logger.warning("FactCheck API key not found")
            return []
        
        try:
            params = {
                'query': claim,
                'key': self.api_key,
                'languageCode': 'en'
            }
            
            response = requests.get(
                self.api_url,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for claim_review in data.get('claims', []):
                for review in claim_review.get('claimReview', []):
                    results.append({
                        'source': 'FactCheck API',
                        'url': review.get('url', ''),
                        'title': review.get('title', ''),
                        'publisher': review.get('publisher', {}).get('name', 'Unknown'),
                        'rating': review.get('textualRating', 'Unknown'),
                        'claim_text': claim_review.get('text', ''),
                        'trust_score': 0.9  # High trust for official fact-checks
                    })
            
            logger.info(f"FactCheck API returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"FactCheck API error: {e}")
            return []


class NewsCheckAPI:
    """Interface to NewsCheck API."""
    
    def __init__(self):
        self.api_key = os.getenv("NEWSCHECK_API_KEY")
        # Note: This is a placeholder URL - replace with actual NewsCheck API endpoint
        self.api_url = "https://api.newscheck.com/v1/verify"
    
    @cache_result(ttl_seconds=604800)  # Cache for 7 days
    def check_claim(self, claim: str) -> List[Dict]:
        """
        Check claim against NewsCheck database.
        
        Args:
            claim: Claim text to check
            
        Returns:
            List of news check results
        """
        if not self.api_key:
            logger.warning("NewsCheck API key not found")
            return []
        
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'query': claim,
                'language': 'en'
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('results', []):
                results.append({
                    'source': 'NewsCheck API',
                    'url': item.get('url', ''),
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', 'Unknown'),
                    'verdict': item.get('verdict', 'Unknown'),
                    'confidence': item.get('confidence', 0.5),
                    'trust_score': 0.85
                })
            
            logger.info(f"NewsCheck API returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"NewsCheck API error: {e}")
            return []


def check_claims_with_apis(claims: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Check claims against FactCheck and NewsCheck APIs.
    
    Args:
        claims: List of claim dicts
        
    Returns:
        Dict mapping claim_id to list of API results
    """
    factcheck_api = FactCheckAPI()
    newscheck_api = NewsCheckAPI()
    
    api_results = {}
    
    for claim in claims:
        claim_id = claim['claim_id']
        claim_text = claim['text']
        
        results = []
        
        # Check FactCheck API
        try:
            fc_results = factcheck_api.check_claim(claim_text)
            results.extend(fc_results)
        except Exception as e:
            logger.error(f"FactCheck API failed for {claim_id}: {e}")
        
        # Check NewsCheck API
        try:
            nc_results = newscheck_api.check_claim(claim_text)
            results.extend(nc_results)
        except Exception as e:
            logger.error(f"NewsCheck API failed for {claim_id}: {e}")
        
        api_results[claim_id] = results
        logger.info(f"API checks returned {len(results)} results for {claim_id}")
    
    return api_results
