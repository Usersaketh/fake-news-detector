"""
SERPER search module for evidence gathering.
Performs multiple queries per claim and returns trusted sources.
"""

import os
import json
import requests
from typing import List, Dict
import logging
from backend.utils import (
    cache_result, 
    retry_on_failure, 
    calculate_trust_score,
    extract_domain,
    serper_limiter
)

logger = logging.getLogger(__name__)


class SerperSearcher:
    """Search for evidence using SERPER API."""
    
    def __init__(self):
        self.api_key = os.getenv("SERPER_API_KEY")
        self.api_url = "https://google.serper.dev/search"
    
    @cache_result(ttl_seconds=86400)  # Cache for 24 hours
    def search_claim(self, claim: str, num_queries: int = 5) -> List[Dict]:
        """
        Search for evidence about a claim using multiple queries.
        
        Args:
            claim: Claim text to search
            num_queries: Number of different queries to perform
            
        Returns:
            List of evidence dicts with url, title, snippet, trust_score
        """
        logger.info(f"Searching for evidence: {claim[:50]}...")
        
        if not self.api_key:
            logger.warning("SERPER API key not found, using mock data")
            return self._mock_search_results(claim)
        
        # Generate multiple search queries
        queries = self._generate_queries(claim)[:num_queries]
        
        all_results = []
        seen_urls = set()
        
        for query in queries:
            try:
                serper_limiter.wait_if_needed()
                results = self._perform_search(query)
                
                # Add results, avoiding duplicates
                for result in results:
                    url = result['url']
                    if url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(result)
                
            except Exception as e:
                logger.error(f"Search failed for query '{query}': {e}")
                continue
        
        # Sort by trust score
        all_results.sort(key=lambda x: x['trust_score'], reverse=True)
        
        # Return top results
        return all_results[:15]
    
    def _generate_queries(self, claim: str) -> List[str]:
        """
        Generate multiple search queries for a claim.
        
        Args:
            claim: Claim text
            
        Returns:
            List of query strings
        """
        # Truncate very long claims
        if len(claim) > 150:
            claim = claim[:150]
        
        queries = [
            f'"{claim}" evidence',
            f'"{claim}" fact check',
            f'{claim} news',
            f'{claim} official statement',
            f'{claim} site:snopes.com OR site:factcheck.org OR site:politifact.com',
            f'{claim} site:reuters.com OR site:apnews.com OR site:bbc.com',
        ]
        
        return queries
    
    @retry_on_failure(max_attempts=2)
    def _perform_search(self, query: str) -> List[Dict]:
        """
        Perform single search query.
        
        Args:
            query: Search query string
            
        Returns:
            List of result dicts
        """
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': query,
            'num': 10
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
        for item in data.get('organic', []):
            url = item.get('link', '')
            domain = extract_domain(url)
            
            # Calculate trust score
            trust_score = calculate_trust_score(
                domain,
                item.get('date', None)
            )
            
            results.append({
                'url': url,
                'title': item.get('title', ''),
                'snippet': item.get('snippet', ''),
                'published': item.get('date', 'Unknown'),
                'domain': domain,
                'trust_score': trust_score
            })
        
        return results
    
    def _mock_search_results(self, claim: str) -> List[Dict]:
        """
        Generate mock search results when API is unavailable.
        
        Args:
            claim: Claim text
            
        Returns:
            Mock results list
        """
        return [
            {
                'url': 'https://snopes.com/fact-check/example',
                'title': f'Fact Check: {claim[:50]}',
                'snippet': 'This claim requires verification. We are investigating.',
                'published': '2024-01-15',
                'domain': 'snopes.com',
                'trust_score': 0.9
            },
            {
                'url': 'https://reuters.com/article/example',
                'title': f'Analysis: {claim[:50]}',
                'snippet': 'Reuters investigates the claims made in recent reports.',
                'published': '2024-01-14',
                'domain': 'reuters.com',
                'trust_score': 0.85
            },
            {
                'url': 'https://bbc.com/news/example',
                'title': f'What we know about {claim[:30]}',
                'snippet': 'BBC Reality Check examines the evidence.',
                'published': '2024-01-13',
                'domain': 'bbc.com',
                'trust_score': 0.8
            }
        ]


def search_for_evidence(claims: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Search for evidence for multiple claims.
    
    Args:
        claims: List of claim dicts with 'claim_id' and 'text'
        
    Returns:
        Dict mapping claim_id to list of evidence dicts
    """
    searcher = SerperSearcher()
    evidence_map = {}
    
    for claim in claims:
        claim_id = claim['claim_id']
        claim_text = claim['text']
        
        evidence = searcher.search_claim(claim_text)
        evidence_map[claim_id] = evidence
        
        logger.info(f"Found {len(evidence)} evidence items for {claim_id}")
    
    return evidence_map
