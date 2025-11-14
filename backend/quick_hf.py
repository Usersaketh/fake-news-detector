"""
HuggingFace quick check module.
Provides fast preliminary classification using multiple lightweight models.
This is displayed separately and NOT considered authoritative.
"""

import os
import json
from typing import Dict, List, Optional
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
    """Quick preliminary check using multiple HuggingFace models."""
    
    def __init__(self):
        self.hf_token = os.getenv("HUGGING_FACE_TOKEN")
        
        # Multiple models for ensemble checking
        self.models = [
            "Pulk17/Fake-News-Detection",
            "vikram71198/distilroberta-base-finetuned-fake-news-detection",
            "jy46604790/Fake-News-Bert-Detect",
            "winterForestStump/Roberta-fake-news-detector"
        ]
        
        self.pipelines = {}
        
        # Initialize all models if transformers available
        if TRANSFORMERS_AVAILABLE:
            for model_name in self.models:
                try:
                    logger.info(f"Loading HuggingFace model: {model_name}")
                    self.pipelines[model_name] = pipeline(
                        "text-classification",
                        model=model_name,
                        token=self.hf_token if self.hf_token else None
                    )
                    logger.info(f"✓ Model loaded: {model_name}")
                except Exception as e:
                    logger.error(f"✗ Failed to load {model_name}: {e}")
                    self.pipelines[model_name] = None
        else:
            logger.warning("transformers not installed, using fallback")
    
    @cache_result(ttl_seconds=3600)
    @retry_on_failure(max_attempts=2)
    def quick_check(self, text: str) -> Dict:
        """
        Perform quick preliminary check on text using multiple HuggingFace models.
        
        Args:
            text: Text to analyze (headline + body preview)
            
        Returns:
            Dict with results from all models and ensemble verdict
        """
        logger.info("Performing HuggingFace quick check with multiple models")
        
        # Truncate text if too long (model has token limits)
        if len(text) > 512:
            text = text[:512]
        
        model_results = []
        
        # Run all available models
        for model_name in self.models:
            if model_name in self.pipelines and self.pipelines[model_name]:
                try:
                    result = self._check_with_model(text, model_name)
                    if result:
                        model_results.append(result)
                except Exception as e:
                    logger.error(f"Model {model_name} failed: {e}")
        
        # If no models worked, use fallback
        if not model_results:
            logger.warning("All HuggingFace models failed, using heuristic fallback")
            return self._fallback_response(text)
        
        # Calculate ensemble verdict
        ensemble_result = self._calculate_ensemble(model_results, text)
        
        return ensemble_result
    
    def _check_with_model(self, text: str, model_name: str) -> Optional[Dict]:
        """
        Check text with a specific model.
        
        Args:
            text: Text to check
            model_name: Name of the model
            
        Returns:
            Result dict or None if failed
        """
        try:
            pipeline_obj = self.pipelines[model_name]
            result = pipeline_obj(text, truncation=True, max_length=512)
            
            # Result format: [{'label': 'LABEL_0' or 'LABEL_1', 'score': 0.95}]
            label_raw = result[0]['label']
            confidence = result[0]['score']
            
            # Map labels - try to handle different label formats
            # Most models: LABEL_0 = Real/True, LABEL_1 = Fake/False
            if label_raw.upper() in ["LABEL_0", "REAL", "TRUE", "LEGIT"]:
                label = "real"
                verdict_text = "Real News"
            elif label_raw.upper() in ["LABEL_1", "FAKE", "FALSE", "UNRELIABLE"]:
                label = "fake"
                verdict_text = "Fake News"
            else:
                # If uncertain, check the score
                label = "fake" if confidence > 0.5 else "real"
                verdict_text = "Fake News" if label == "fake" else "Real News"
            
            return {
                "model": model_name.split('/')[-1],  # Short name
                "full_model": model_name,
                "label": label,
                "confidence": float(confidence),
                "verdict": verdict_text,
                "raw_label": label_raw
            }
            
        except Exception as e:
            logger.error(f"Error checking with {model_name}: {e}")
            return None
    
    def _calculate_ensemble(self, model_results: List[Dict], text: str) -> Dict:
        """
        Calculate ensemble verdict from multiple model results.
        
        Args:
            model_results: List of results from individual models
            text: Original text
            
        Returns:
            Ensemble result dict
        """
        # Count votes
        fake_votes = sum(1 for r in model_results if r['label'] == 'fake')
        real_votes = sum(1 for r in model_results if r['label'] == 'real')
        
        # Calculate average confidence
        avg_confidence = sum(r['confidence'] for r in model_results) / len(model_results)
        
        # Determine ensemble verdict
        if fake_votes > real_votes:
            ensemble_label = "likely-false"
            ensemble_verdict = "Fake News (Ensemble)"
        elif real_votes > fake_votes:
            ensemble_label = "likely-true"
            ensemble_verdict = "Real News (Ensemble)"
        else:
            ensemble_label = "uncertain"
            ensemble_verdict = "Mixed Results"
        
        summary = self._generate_summary(text)
        
        return {
            "summary": summary,
            "label": ensemble_label,
            "confidence": avg_confidence,
            "verdict": ensemble_verdict,
            "model": f"Ensemble ({len(model_results)} models)",
            "individual_results": model_results,
            "vote_breakdown": {
                "fake": fake_votes,
                "real": real_votes,
                "total": len(model_results)
            }
        }
    
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
