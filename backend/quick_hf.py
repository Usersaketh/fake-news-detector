"""
HuggingFace quick check module.
Provides fast preliminary classification using a lightweight model.
This is displayed separately and NOT considered authoritative.
"""

import os
import json
from typing import Dict, Optional
import logging
from backend.utils import cache_result, retry_on_failure

logger = logging.getLogger(__name__)

# Try to import transformers
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers package not available. Install with: pip install transformers torch")


class HuggingFaceQuickCheck:
    """Quick preliminary check using HuggingFace models."""
    
    def __init__(self):
        self.hf_token = os.getenv("HUGGING_FACE_TOKEN")
        self.model_name = "Pulk17/Fake-News-Detection"
        self.pipeline = None
        
        # Initialize pipeline if transformers available
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading HuggingFace model: {self.model_name}")
                self.pipeline = pipeline(
                    "text-classification",
                    model=self.model_name,
                    token=self.hf_token if self.hf_token else None
                )
                logger.info("HuggingFace model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load HuggingFace model: {e}")
                self.pipeline = None
        else:
            logger.warning("transformers not installed, using fallback")
    
    @cache_result(ttl_seconds=3600)
    @retry_on_failure(max_attempts=2)
    def quick_check(self, text: str) -> Dict:
        """
        Perform quick preliminary check on text using HuggingFace model.
        
        Args:
            text: Text to analyze (headline + body preview)
            
        Returns:
            Dict with summary, label, and confidence
        """
        logger.info("Performing HuggingFace quick check with Fake-News-Detection model")
        
        # Truncate text if too long (model has token limits)
        if len(text) > 512:
            text = text[:512]
        
        # Use HuggingFace pipeline if available
        if self.pipeline:
            try:
                # Run classification
                result = self.pipeline(text, truncation=True, max_length=512)
                
                # Result format: [{'label': 'LABEL_0' or 'LABEL_1', 'score': 0.95}]
                label_raw = result[0]['label']
                confidence = result[0]['score']
                
                # Map labels (LABEL_0 = Real, LABEL_1 = Fake for this model)
                if label_raw == "LABEL_0":
                    label = "likely-true"
                    verdict_text = "Real News"
                else:
                    label = "likely-false"
                    verdict_text = "Fake News"
                
                # Generate simple summary (first sentence)
                summary = self._generate_summary(text)
                
                result_dict = {
                    "summary": summary,
                    "label": label,
                    "confidence": float(confidence),
                    "raw_label": label_raw,
                    "verdict": verdict_text,
                    "model": "Pulk17/Fake-News-Detection"
                }
                
                logger.info(f"HF quick check complete: {label} (confidence: {confidence:.2f})")
                return result_dict
                
            except Exception as e:
                logger.error(f"HF pipeline failed: {e}")
                return self._fallback_response(text)
        else:
            # Fallback if model not loaded
            logger.warning("HuggingFace model not available, using heuristic fallback")
            return self._fallback_response(text)
    
    def _generate_summary(self, text: str) -> str:
        """
        Generate simple summary from text.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summary string
        """
        # Get first sentence or first 100 chars
        sentences = text.split('.')
        if sentences:
            summary = sentences[0].strip()
            if len(summary) > 100:
                summary = summary[:100] + "..."
            return summary
        return text[:100] + "..." if len(text) > 100 else text
    
    def _fallback_response(self, text: str) -> Dict:
        """
        Generate fallback response when model fails.
        
        Args:
            text: Original text
            
        Returns:
            Fallback response dict
        """
        # Simple heuristic-based fallback
        text_lower = text.lower()
        
        # Look for sensational words (fake news indicators)
        sensational_words = [
            'shocking', 'unbelievable', 'secret', 'they dont want you to know',
            'miracle', 'cure', 'conspiracy', 'hoax', 'breaking', 'exclusive',
            'urgent', 'alert', 'must see', 'must read'
        ]
        
        sensational_count = sum(1 for word in sensational_words if word in text_lower)
        
        # Look for credibility indicators
        credible_words = ['study', 'research', 'according to', 'report', 'published']
        credible_count = sum(1 for word in credible_words if word in text_lower)
        
        if sensational_count >= 3:
            label = "likely-false"
            confidence = 0.65
            verdict = "Possibly Fake (heuristic)"
        elif credible_count >= 2 and sensational_count == 0:
            label = "likely-true"
            confidence = 0.55
            verdict = "Possibly Real (heuristic)"
        else:
            label = "uncertain"
            confidence = 0.4
            verdict = "Uncertain (heuristic)"
        
        summary = self._generate_summary(text)
        
        return {
            "summary": summary,
            "label": label,
            "confidence": confidence,
            "verdict": verdict,
            "note": "⚠️ Fallback heuristic used (HuggingFace model not available)",
            "model": "heuristic_fallback"
        }


def perform_quick_check(text: str) -> Dict:
    """
    Perform quick check on text.
    
    Args:
        text: Text to check
        
    Returns:
        Quick check result dict
    """
    checker = HuggingFaceQuickCheck()
    return checker.quick_check(text)
