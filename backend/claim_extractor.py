"""
Claim extraction module.
Extracts verifiable factual claims from article text.
"""

import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class ClaimExtractor:
    """Extract verifiable claims from text."""
    
    def __init__(self):
        # Patterns that indicate factual claims
        self.claim_indicators = [
            r'\d+%',  # Percentages
            r'\d+[\s,]+(people|deaths|cases|dollars|million|billion|thousand)',  # Statistics
            r'(said|stated|announced|declared|confirmed|reported)\s+that',  # Reported statements
            r'(is|was|are|were|has|have|will)\s+(the|a)\s+(first|largest|biggest|most|highest)',  # Superlatives
            r'according to',  # Citations
        ]
    
    def extract_claims(self, headline: str, body: str, max_claims: int = 5) -> List[Dict]:
        """
        Extract verifiable claims from article.
        
        Args:
            headline: Article headline
            body: Article body text
            max_claims: Maximum number of claims to extract (1 = whole article mode)
            
        Returns:
            List of claim dicts with claim_id, text, and source
        """
        claims = []
        
        # If max_claims is 1, treat entire article as single claim
        if max_claims == 1:
            # Combine headline and first 2-3 sentences of body
            body_preview = '. '.join(self._split_into_sentences(body)[:3]) if body else ""
            full_text = f"{headline}. {body_preview}".strip()
            
            claims.append({
                "claim_id": "claim_0",
                "text": full_text[:500],  # Limit to 500 chars for API efficiency
                "source": "article",
                "priority": 1
            })
            logger.info(f"Extracted 1 claim (whole article mode)")
            return claims
        
        # Multi-claim mode: Extract headline + body claims
        # Always include the headline as the primary claim
        if headline and len(headline) > 10:
            claims.append({
                "claim_id": f"claim_0",
                "text": headline.strip(),
                "source": "headline",
                "priority": 1
            })
        
        # Extract sentences from body
        sentences = self._split_into_sentences(body)
        
        # Score and rank sentences by claim likelihood
        scored_sentences = []
        for sentence in sentences:
            score = self._score_claim_likelihood(sentence)
            if score > 0:
                scored_sentences.append((sentence, score))
        
        # Sort by score descending
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Take top sentences as claims
        for i, (sentence, score) in enumerate(scored_sentences[:max_claims-1]):  # -1 for headline
            claims.append({
                "claim_id": f"claim_{i+1}",
                "text": sentence.strip(),
                "source": "body",
                "priority": 2 if i == 0 else 3,
                "score": score
            })
        
        logger.info(f"Extracted {len(claims)} claims")
        return claims
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with NLTK)
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Filter out very short or very long sentences
        sentences = [s for s in sentences if 20 < len(s) < 300]
        
        return sentences
    
    def _score_claim_likelihood(self, sentence: str) -> float:
        """
        Score how likely a sentence is to contain a verifiable claim.
        
        Args:
            sentence: Sentence to score
            
        Returns:
            Score (higher = more likely to be a claim)
        """
        score = 0.0
        sentence_lower = sentence.lower()
        
        # Check for claim indicators
        for pattern in self.claim_indicators:
            if re.search(pattern, sentence_lower):
                score += 1.0
        
        # Boost for numbers and dates
        if re.search(r'\d+', sentence):
            score += 0.5
        
        # Boost for proper nouns (capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', sentence)
        if len(proper_nouns) >= 2:
            score += 0.3
        
        # Boost for quotations
        if '"' in sentence or "'" in sentence:
            score += 0.4
        
        # Penalize questions
        if '?' in sentence:
            score -= 0.5
        
        # Penalize opinion words
        opinion_words = ['think', 'believe', 'feel', 'opinion', 'may', 'might', 'could', 'perhaps']
        if any(word in sentence_lower for word in opinion_words):
            score -= 0.3
        
        # Penalize very short sentences
        if len(sentence) < 50:
            score -= 0.2
        
        return max(0.0, score)
    
    def merge_similar_claims(self, claims: List[Dict], similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Merge very similar claims to avoid redundancy.
        
        Args:
            claims: List of claim dicts
            similarity_threshold: Threshold for merging (0-1)
            
        Returns:
            Deduplicated list of claims
        """
        if len(claims) <= 1:
            return claims
        
        merged = []
        skip_indices = set()
        
        for i, claim1 in enumerate(claims):
            if i in skip_indices:
                continue
            
            # Check against remaining claims
            for j in range(i + 1, len(claims)):
                if j in skip_indices:
                    continue
                
                claim2 = claims[j]
                similarity = self._calculate_similarity(claim1['text'], claim2['text'])
                
                if similarity >= similarity_threshold:
                    # Merge claims (keep higher priority one)
                    if claim1.get('priority', 999) <= claim2.get('priority', 999):
                        skip_indices.add(j)
                    else:
                        skip_indices.add(i)
                        break
            
            if i not in skip_indices:
                merged.append(claim1)
        
        return merged
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)


def extract_claims(headline: str, body: str, max_claims: int = 5) -> List[Dict]:
    """
    Extract claims from article text.
    
    Args:
        headline: Article headline
        body: Article body
        max_claims: Maximum number of claims to extract
        
    Returns:
        List of claim dictionaries
    """
    extractor = ClaimExtractor()
    claims = extractor.extract_claims(headline, body, max_claims)
    claims = extractor.merge_similar_claims(claims)
    return claims
