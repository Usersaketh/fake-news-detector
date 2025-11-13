"""
Aggregator module.
Combines individual claim verdicts into overall article verdict.
"""

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class VerdictAggregator:
    """Aggregate claim verdicts into overall assessment."""
    
    def aggregate(
        self,
        verdicts: List[Dict],
        evidence_map: Dict[str, List[Dict]]
    ) -> Dict:
        """
        Aggregate claim verdicts into overall verdict.
        
        Args:
            verdicts: List of claim verdict dicts
            evidence_map: Dict mapping claim_id to evidence
            
        Returns:
            Overall verdict dict
        """
        logger.info(f"Aggregating {len(verdicts)} claim verdicts")
        
        if not verdicts:
            return self._empty_response()
        
        # Count verdicts by type
        verdict_counts = {
            'true': 0,
            'false': 0,
            'mixed': 0,
            'insufficient': 0
        }
        
        confidence_sum = 0.0
        high_confidence_false = []
        
        for verdict in verdicts:
            v_type = verdict.get('verdict', 'insufficient')
            confidence = verdict.get('confidence', 0.0)
            
            verdict_counts[v_type] += 1
            confidence_sum += confidence
            
            # Track high-confidence false claims
            if v_type == 'false' and confidence > 0.85:
                high_confidence_false.append(verdict)
        
        # Apply aggregation rules
        overall_verdict, overall_confidence = self._apply_rules(
            verdict_counts,
            high_confidence_false,
            confidence_sum,
            len(verdicts)
        )
        
        # Find primary source (highest trust score evidence)
        primary_source = self._find_primary_source(verdicts, evidence_map)
        
        return {
            'overall_verdict': overall_verdict,
            'overall_confidence': overall_confidence,
            'primary_source': primary_source,
            'verdict_breakdown': verdict_counts,
            'total_claims': len(verdicts),
            'claims': verdicts
        }
    
    def _apply_rules(
        self,
        counts: Dict[str, int],
        high_conf_false: List[Dict],
        confidence_sum: float,
        total: int
    ) -> tuple:
        """
        Apply aggregation rules to determine overall verdict.
        
        Args:
            counts: Verdict type counts
            high_conf_false: List of high-confidence false verdicts
            confidence_sum: Sum of all confidences
            total: Total number of verdicts
            
        Returns:
            Tuple of (overall_verdict, overall_confidence)
        """
        avg_confidence = confidence_sum / total if total > 0 else 0.0
        
        # Count only substantive verdicts (exclude insufficient)
        substantive_total = counts['true'] + counts['false'] + counts['mixed']
        
        # Rule 1: Any high-confidence false → article is false
        if high_conf_false:
            return ('false', min(0.95, avg_confidence + 0.1))
        
        # Rule 2: Majority false (even without high confidence) → likely false
        if substantive_total > 0 and counts['false'] > substantive_total / 2:
            return ('false', avg_confidence * 0.9)
        
        # Rule 3: Any true verdicts and no false → article is true
        # (Don't let insufficient claims drag it down)
        if counts['true'] > 0 and counts['false'] == 0:
            # If we have at least one true and no false, lean towards true
            if substantive_total > 0 and counts['true'] >= substantive_total / 2:
                return ('true', avg_confidence)
            elif counts['true'] >= 1 and substantive_total <= 2:
                # If only 1-2 substantive verdicts and at least one is true
                return ('true', avg_confidence * 0.9)
        
        # Rule 4: Majority true among substantive verdicts → true
        if substantive_total > 0 and counts['true'] > substantive_total / 2:
            return ('true', avg_confidence)
        
        # Rule 5: Conflicting verdicts (both true and false) → mixed
        if counts['true'] > 0 and counts['false'] > 0:
            return ('mixed', avg_confidence * 0.8)
        
        # Rule 6: Several mixed verdicts → mixed
        if counts['mixed'] >= 2:
            return ('mixed', avg_confidence * 0.8)
        
        # Rule 7: All insufficient → overall insufficient
        if substantive_total == 0:
            return ('insufficient', avg_confidence * 0.7)
        
        # Default: insufficient (but this should rarely be reached)
        return ('insufficient', avg_confidence * 0.6)
    
    def _find_primary_source(
        self,
        verdicts: List[Dict],
        evidence_map: Dict[str, List[Dict]]
    ) -> Dict:
        """
        Find the most authoritative source cited.
        
        Args:
            verdicts: List of verdicts
            evidence_map: Evidence mapping
            
        Returns:
            Primary source dict with url and title
        """
        best_source = None
        best_score = 0.0
        
        for verdict in verdicts:
            claim_id = verdict.get('claim_id')
            if not claim_id:
                continue
            
            # Get evidence for this claim
            evidence = evidence_map.get(claim_id, [])
            
            # Find highest trust score
            for ev in evidence:
                trust_score = ev.get('trust_score', 0.0)
                if trust_score > best_score:
                    best_score = trust_score
                    best_source = {
                        'url': ev.get('url', ''),
                        'title': ev.get('title', 'Unknown'),
                        'trust_score': trust_score
                    }
        
        if not best_source:
            return {
                'url': '',
                'title': 'No authoritative source found',
                'trust_score': 0.0
            }
        
        return best_source
    
    def _empty_response(self) -> Dict:
        """Generate empty response when no verdicts."""
        return {
            'overall_verdict': 'insufficient',
            'overall_confidence': 0.0,
            'primary_source': {
                'url': '',
                'title': 'No evidence available',
                'trust_score': 0.0
            },
            'verdict_breakdown': {
                'true': 0,
                'false': 0,
                'mixed': 0,
                'insufficient': 0
            },
            'total_claims': 0,
            'claims': []
        }


def aggregate_verdicts(
    verdicts: List[Dict],
    evidence_map: Dict[str, List[Dict]]
) -> Dict:
    """
    Aggregate claim verdicts into overall assessment.
    
    Args:
        verdicts: List of claim verdicts
        evidence_map: Evidence mapping
        
    Returns:
        Overall verdict dict
    """
    aggregator = VerdictAggregator()
    return aggregator.aggregate(verdicts, evidence_map)
