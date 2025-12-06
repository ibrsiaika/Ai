"""
Evaluation metrics for India's GPT model.
Includes perplexity, BLEU, ROUGE, and other NLG metrics.
"""

import torch
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """Container for metrics results."""
    perplexity: float
    bleu: float
    rouge1: Dict[str, float]
    rouge2: Dict[str, float]
    rougeL: Dict[str, float]
    loss: Optional[float] = None


class LanguageModelingMetrics:
    """Metrics for language modeling tasks."""
    
    @staticmethod
    def perplexity(loss: float) -> float:
        """
        Calculate perplexity from loss.
        
        Args:
            loss: Cross-entropy loss
        
        Returns:
            Perplexity
        """
        return math.exp(loss)
    
    @staticmethod
    def token_accuracy(predictions: torch.Tensor,
                      targets: torch.Tensor,
                      ignore_index: int = -100) -> float:
        """
        Calculate token-level accuracy.
        
        Args:
            predictions: Predicted token IDs (batch_size, seq_len)
            targets: Target token IDs (batch_size, seq_len)
            ignore_index: Index to ignore (e.g., padding)
        
        Returns:
            Accuracy
        """
        # Flatten tensors
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)
        
        # Create mask for valid tokens
        mask = target_flat != ignore_index
        
        # Calculate accuracy
        correct = (pred_flat[mask] == target_flat[mask]).sum().item()
        total = mask.sum().item()
        
        return correct / total if total > 0 else 0.0


class BLEUScore:
    """BLEU score calculation."""
    
    @staticmethod
    def _get_ngrams(segment: List[int], max_order: int) -> Counter:
        """Extract n-grams from segment."""
        ngram_counts = Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i + order])
                ngram_counts[ngram] += 1
        return ngram_counts
    
    @staticmethod
    def compute_bleu(reference_corpus: List[List[int]],
                    translation_corpus: List[List[int]],
                    max_order: int = 4,
                    smooth: bool = False) -> float:
        """
        Compute BLEU score.
        
        Args:
            reference_corpus: List of reference token sequences
            translation_corpus: List of predicted token sequences
            max_order: Maximum n-gram order
            smooth: Whether to apply smoothing
        
        Returns:
            BLEU score
        """
        matches_by_order = [0] * max_order
        possible_matches_by_order = [0] * max_order
        reference_length = 0
        translation_length = 0
        
        for references, translation in zip(reference_corpus, translation_corpus):
            reference_length += min(len(r) for r in (references if isinstance(references[0], list) else [references]))
            translation_length += len(translation)
            
            # Handle both single reference and multiple references
            if not isinstance(references[0], (list, tuple)):
                references = [references]
            
            # Get reference n-grams
            merged_ref_ngram_counts = Counter()
            for reference in references:
                reference_ngrams = BLEUScore._get_ngrams(reference, max_order)
                for ngram in reference_ngrams:
                    merged_ref_ngram_counts[ngram] = max(
                        merged_ref_ngram_counts[ngram],
                        reference_ngrams[ngram]
                    )
            
            # Get translation n-grams
            translation_ngrams = BLEUScore._get_ngrams(translation, max_order)
            
            # Count matches
            for ngram in translation_ngrams:
                matches = min(translation_ngrams[ngram], merged_ref_ngram_counts.get(ngram, 0))
                matches_by_order[len(ngram) - 1] += matches
            
            # Count possible matches
            for order in range(1, max_order + 1):
                possible_matches = len(translation) - order + 1
                if possible_matches > 0:
                    possible_matches_by_order[order - 1] += possible_matches
        
        # Calculate precisions
        precisions = [0] * max_order
        for i in range(0, max_order):
            if smooth:
                precisions[i] = ((matches_by_order[i] + 1.0) /
                                (possible_matches_by_order[i] + 1.0))
            else:
                if possible_matches_by_order[i] > 0:
                    precisions[i] = (float(matches_by_order[i]) /
                                    possible_matches_by_order[i])
                else:
                    precisions[i] = 0.0
        
        # Calculate brevity penalty
        if not precisions:
            bp = 0.0
        else:
            ratio = float(translation_length) / reference_length
            if ratio > 1.0:
                bp = 1.0
            else:
                bp = math.exp(1 - 1.0 / ratio) if ratio > 0 else 0.0
        
        # Calculate BLEU
        geometric_mean = math.exp(sum((1.0 / max_order) * math.log(p) for p in precisions if p > 0))
        
        return bp * geometric_mean


class ROUGEScore:
    """ROUGE score calculation."""
    
    @staticmethod
    def _get_tokens(text: str) -> List[str]:
        """Tokenize text."""
        return text.split()
    
    @staticmethod
    def _rouge_n(reference: str, hypothesis: str, n: int) -> Tuple[float, float, float]:
        """
        Calculate ROUGE-N score.
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
            n: N-gram order
        
        Returns:
            Tuple of (recall, precision, f1)
        """
        ref_tokens = ROUGEScore._get_tokens(reference)
        hyp_tokens = ROUGEScore._get_tokens(hypothesis)
        
        # Get n-grams
        ref_ngrams = Counter()
        for i in range(len(ref_tokens) - n + 1):
            ngram = tuple(ref_tokens[i:i + n])
            ref_ngrams[ngram] += 1
        
        hyp_ngrams = Counter()
        for i in range(len(hyp_tokens) - n + 1):
            ngram = tuple(hyp_tokens[i:i + n])
            hyp_ngrams[ngram] += 1
        
        # Count matches
        matches = 0
        for ngram in hyp_ngrams:
            matches += min(hyp_ngrams[ngram], ref_ngrams.get(ngram, 0))\n        # Calculate recall and precision
        recall = matches / sum(ref_ngrams.values()) if ref_ngrams else 0.0
        precision = matches / sum(hyp_ngrams.values()) if hyp_ngrams else 0.0
        
        # F1 score
        if recall + precision == 0:
            f1 = 0.0
        else:
            f1 = 2 * (recall * precision) / (recall + precision)
        
        return recall, precision, f1
    
    @staticmethod
    def _rouge_l(reference: str, hypothesis: str) -> Tuple[float, float, float]:
        """
        Calculate ROUGE-L score (LCS-based).
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
        
        Returns:
            Tuple of (recall, precision, f1)
        """
        ref_tokens = ROUGEScore._get_tokens(reference)
        hyp_tokens = ROUGEScore._get_tokens(hypothesis)
        
        # Calculate LCS length
        lcs_len = ROUGEScore._lcs(ref_tokens, hyp_tokens)
        
        recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
        precision = lcs_len / len(hyp_tokens) if hyp_tokens else 0.0
        
        if recall + precision == 0:
            f1 = 0.0
        else:
            f1 = 2 * (recall * precision) / (recall + precision)
        
        return recall, precision, f1
    
    @staticmethod
    def _lcs(s1: List[str], s2: List[str]) -> int:
        """Calculate longest common subsequence length."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    @staticmethod
    def compute_rouge(references: List[str],
                     hypotheses: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compute ROUGE scores.
        
        Args:
            references: List of reference texts
            hypotheses: List of hypothesis texts
        
        Returns:
            Dictionary of ROUGE scores
        """
        rouge1_recalls = []
        rouge1_precisions = []
        rouge1_f1s = []
        
        rouge2_recalls = []
        rouge2_precisions = []
        rouge2_f1s = []
        
        rougeL_recalls = []
        rougeL_precisions = []
        rougeL_f1s = []
        
        for ref, hyp in zip(references, hypotheses):
            # ROUGE-1
            r1_recall, r1_precision, r1_f1 = ROUGEScore._rouge_n(ref, hyp, 1)
            rouge1_recalls.append(r1_recall)
            rouge1_precisions.append(r1_precision)
            rouge1_f1s.append(r1_f1)
            
            # ROUGE-2
            r2_recall, r2_precision, r2_f1 = ROUGEScore._rouge_n(ref, hyp, 2)
            rouge2_recalls.append(r2_recall)
            rouge2_precisions.append(r2_precision)
            rouge2_f1s.append(r2_f1)
            
            # ROUGE-L
            rl_recall, rl_precision, rl_f1 = ROUGEScore._rouge_l(ref, hyp)
            rougeL_recalls.append(rl_recall)
            rougeL_precisions.append(rl_precision)
            rougeL_f1s.append(rl_f1)
        
        return {
            'rouge1': {
                'recall': np.mean(rouge1_recalls),
                'precision': np.mean(rouge1_precisions),
                'f1': np.mean(rouge1_f1s)
            },
            'rouge2': {
                'recall': np.mean(rouge2_recalls),
                'precision': np.mean(rouge2_precisions),
                'f1': np.mean(rouge2_f1s)
            },
            'rougeL': {
                'recall': np.mean(rougeL_recalls),
                'precision': np.mean(rougeL_precisions),
                'f1': np.mean(rougeL_f1s)
            }
        }


class MetricsCalculator:
    """Main metrics calculator."""
    
    def __init__(self, tokenizer):
        """
        Initialize calculator.
        
        Args:
            tokenizer: Tokenizer for decoding
        """
        self.tokenizer = tokenizer
    
    def calculate_metrics(self,
                         predictions: List[str],
                         references: List[str],
                         loss: Optional[float] = None) -> MetricsResult:
        """
        Calculate all metrics.
        
        Args:
            predictions: Predicted texts
            references: Reference texts
            loss: Model loss for perplexity
        
        Returns:
            MetricsResult with all computed metrics
        """
        # BLEU
        pred_tokens = [self.tokenizer.encode(p) for p in predictions]
        ref_tokens = [[self.tokenizer.encode(r)] for r in references]
        bleu = BLEUScore.compute_bleu(ref_tokens, pred_tokens)
        
        # ROUGE
        rouge_scores = ROUGEScore.compute_rouge(references, predictions)
        
        # Perplexity
        perplexity = LanguageModelingMetrics.perplexity(loss) if loss else 0.0
        
        return MetricsResult(
            perplexity=perplexity,
            bleu=bleu,
            rouge1=rouge_scores['rouge1'],
            rouge2=rouge_scores['rouge2'],
            rougeL=rouge_scores['rougeL'],
            loss=loss
        )


def create_metrics_calculator(tokenizer) -> MetricsCalculator:
    """Create metrics calculator."""
    return MetricsCalculator(tokenizer)
