""" 
Semantic Similarity Score Evaluation to create embeddings of LLM generated responses to measure cosine similarities of responses
Functions are called in the main evaluation script
"""

import json
import logging
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from ..llm_integration.extraction_models import parse_llm_json_response

logger = logging.getLogger(__name__)


class SemScoreEvaluator:
    """Evaluates semantic similarity between consumer explanations using sentence transformers."""
    
    _model_cache = {}  # Class-level cache for model instances
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        """
        Initialize SemScore evaluator.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        try:
            # Check cache first to avoid reloading models
            if model_name in self._model_cache:
                self.model = self._model_cache[model_name]
                logger.debug(f"🔄 Reusing cached SemScore model: {model_name}")
            else:
                self.model = SentenceTransformer(model_name)
                self._model_cache[model_name] = self.model
                logger.info(f"✅ SemScore initialized with model: {model_name}")
            
            self.model_name = model_name
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            raise
    
    def extract_consumer_explanation(self, explanation_text: str) -> str:
        """Extract consumer explanation from LLM response."""
        try:
            parsed_response = parse_llm_json_response(explanation_text)
            if parsed_response and parsed_response.consumer_explanation:
                return parsed_response.consumer_explanation.strip()
        except Exception as e:
            logger.warning(f"Failed to parse explanation, using raw text: {e}")
        
        return explanation_text.strip()
    
    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute sentence embeddings for a list of texts."""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}")
            raise
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix between embeddings."""
        return cosine_similarity(embeddings)
    
    def evaluate_explanation_similarity(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate semantic similarity between consumer explanations with granular analysis.
        
        Args:
            explanations: List of explanation dictionaries
            
        Returns:
            Dictionary with similarity analysis results by LLM and prediction outcome
        """
        logger.info(f"🔍 Computing semantic similarity for {len(explanations)} explanations")
        
        # Extract consumer explanations with prediction outcomes
        consumer_texts = []
        explanation_metadata = []
        
        for exp in explanations:
            consumer_text = self.extract_consumer_explanation(exp.get('explanation_text', ''))
            if consumer_text:
                consumer_texts.append(consumer_text)
                explanation_metadata.append({
                    'prediction_id': exp.get('prediction_id'),
                    'llm_name': exp.get('llm_name'),
                    'cluster_id': exp.get('cluster_id'),
                    'explanation_id': exp.get('explanation_id'),
                    'regeneration_number': exp.get('regeneration_number'),
                    'prediction_outcome': exp.get('prediction_outcome', 'unknown')  # approved/declined
                })
        
        if len(consumer_texts) < 2:
            return {
                'error': 'Insufficient explanations for similarity analysis',
                'num_explanations': len(consumer_texts)
            }
        
        # Compute embeddings
        logger.info("Computing sentence embeddings...")
        embeddings = self.compute_embeddings(consumer_texts)
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        
        # Analyze similarities with granular breakdown
        analysis = self._analyze_similarities_granular(similarity_matrix, explanation_metadata)
        
        # Add metadata
        analysis.update({
            'model_used': self.model_name,
            'num_explanations': len(consumer_texts),
            'embedding_dimension': embeddings.shape[1]
        })
        
        logger.info("✅ Semantic similarity analysis complete")
        return analysis
    
    def _analyze_similarities_granular(self, similarity_matrix: np.ndarray, 
                                     metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze similarity patterns with granular breakdown by LLM and prediction outcome."""
        
        # Overall analysis (keeping original method for backward compatibility)
        overall_analysis = self._analyze_similarities(similarity_matrix, metadata)
        
        # Granular analysis by LLM
        llm_analyses = {}
        llm_names = set(m['llm_name'] for m in metadata)
        
        for llm_name in llm_names:
            # Filter explanations for this LLM
            llm_indices = [i for i, m in enumerate(metadata) if m['llm_name'] == llm_name]
            if len(llm_indices) < 2:
                continue
                
            # Extract sub-matrix for this LLM
            llm_similarity_matrix = similarity_matrix[np.ix_(llm_indices, llm_indices)]
            llm_metadata = [metadata[i] for i in llm_indices]
            
            # Analyze this LLM's explanations
            llm_analysis = self._analyze_similarities(llm_similarity_matrix, llm_metadata)
            llm_analyses[llm_name] = llm_analysis
        
        # Granular analysis by prediction outcome
        outcome_analyses = {}
        outcomes = set(m['prediction_outcome'] for m in metadata)
        
        for outcome in outcomes:
            # Filter explanations for this outcome
            outcome_indices = [i for i, m in enumerate(metadata) if m['prediction_outcome'] == outcome]
            if len(outcome_indices) < 2:
                continue
                
            # Extract sub-matrix for this outcome
            outcome_similarity_matrix = similarity_matrix[np.ix_(outcome_indices, outcome_indices)]
            outcome_metadata = [metadata[i] for i in outcome_indices]
            
            # Analyze this outcome's explanations
            outcome_analysis = self._analyze_similarities(outcome_similarity_matrix, outcome_metadata)
            outcome_analyses[outcome] = outcome_analysis
        
        # Analysis by LLM AND outcome (most meaningful)
        llm_outcome_analyses = {}
        for llm_name in llm_names:
            llm_outcome_analyses[llm_name] = {}
            for outcome in outcomes:
                # Filter explanations for this LLM AND outcome
                llm_outcome_indices = [i for i, m in enumerate(metadata) 
                                     if m['llm_name'] == llm_name and m['prediction_outcome'] == outcome]
                if len(llm_outcome_indices) < 2:
                    continue
                    
                # Extract sub-matrix for this LLM+outcome combination
                llm_outcome_similarity_matrix = similarity_matrix[np.ix_(llm_outcome_indices, llm_outcome_indices)]
                llm_outcome_metadata = [metadata[i] for i in llm_outcome_indices]
                
                # Analyze this LLM+outcome combination
                llm_outcome_analysis = self._analyze_similarities(llm_outcome_similarity_matrix, llm_outcome_metadata)
                llm_outcome_analyses[llm_name][outcome] = llm_outcome_analysis
        
        # Combine all analyses
        return {
            'overall': overall_analysis,
            'by_llm': llm_analyses,
            'by_outcome': outcome_analyses,
            'by_llm_and_outcome': llm_outcome_analyses
        }
    
    def _analyze_similarities(self, similarity_matrix: np.ndarray, 
                            metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze similarity patterns from the similarity matrix."""
        
        n = len(similarity_matrix)
        
        # 1. Random baseline (all pairs)
        upper_triangle_mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
        all_similarities = similarity_matrix[upper_triangle_mask]
        
        random_baseline = {
            'mean': float(np.mean(all_similarities)),
            'std': float(np.std(all_similarities)),
            'median': float(np.median(all_similarities)),
            'min': float(np.min(all_similarities)),
            'max': float(np.max(all_similarities)),
            'count': len(all_similarities)
        }
        
        # 2. Same-instance similarity (different regenerations of same prediction)
        same_instance_similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                if (metadata[i]['prediction_id'] == metadata[j]['prediction_id'] and
                    metadata[i]['regeneration_number'] != metadata[j]['regeneration_number']):
                    same_instance_similarities.append(similarity_matrix[i, j])
        
        same_instance_stats = self._compute_stats(same_instance_similarities, "same_instance")
        
        # 3. Intra-cluster similarity (same cluster, different instances)
        intra_cluster_similarities = []
        cluster_similarities = {}
        
        for i in range(n):
            for j in range(i + 1, n):
                cluster_i = metadata[i]['cluster_id']
                cluster_j = metadata[j]['cluster_id']
                
                if (cluster_i is not None and cluster_j is not None and 
                    cluster_i == cluster_j and 
                    metadata[i]['prediction_id'] != metadata[j]['prediction_id']):
                    
                    sim = similarity_matrix[i, j]
                    intra_cluster_similarities.append(sim)
                    
                    # Track by cluster
                    if cluster_i not in cluster_similarities:
                        cluster_similarities[cluster_i] = []
                    cluster_similarities[cluster_i].append(sim)
        
        intra_cluster_stats = self._compute_stats(intra_cluster_similarities, "intra_cluster")
        
        # 4. Inter-cluster similarity (different clusters)
        inter_cluster_similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                cluster_i = metadata[i]['cluster_id']
                cluster_j = metadata[j]['cluster_id']
                
                if (cluster_i is not None and cluster_j is not None and 
                    cluster_i != cluster_j and 
                    metadata[i]['prediction_id'] != metadata[j]['prediction_id']):
                    inter_cluster_similarities.append(similarity_matrix[i, j])
        
        inter_cluster_stats = self._compute_stats(inter_cluster_similarities, "inter_cluster")
        
        # 5. Per-cluster statistics
        cluster_stats = {}
        for cluster_id, similarities in cluster_similarities.items():
            cluster_stats[str(cluster_id)] = self._compute_stats(similarities, f"cluster_{cluster_id}")
        
        # 6. Consistency score (how much higher intra-cluster is vs inter-cluster)
        consistency_score = None
        if intra_cluster_stats['count'] > 0 and inter_cluster_stats['count'] > 0:
            consistency_score = (intra_cluster_stats['mean'] - inter_cluster_stats['mean']) / random_baseline['std']
        
        return {
            'random_baseline': random_baseline,
            'same_instance_similarity': same_instance_stats,
            'intra_cluster_similarity': intra_cluster_stats,
            'inter_cluster_similarity': inter_cluster_stats,
            'per_cluster_similarity': cluster_stats,
            'consistency_score': consistency_score,
            'similarity_hierarchy': self._create_similarity_hierarchy(
                same_instance_stats, intra_cluster_stats, inter_cluster_stats, random_baseline
            )
        }
    
    def _compute_stats(self, similarities: List[float], category: str) -> Dict[str, Any]:
        """Compute statistics for a list of similarities."""
        if not similarities:
            return {
                'mean': None,
                'std': None,
                'median': None,
                'min': None,
                'max': None,
                'count': 0,
                'category': category
            }
        
        return {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'median': float(np.median(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'count': len(similarities),
            'category': category
        }
    
    def _create_similarity_hierarchy(self, same_instance: Dict, intra_cluster: Dict, 
                                   inter_cluster: Dict, random_baseline: Dict) -> Dict[str, Any]:
        """Create a hierarchy ranking of similarity types."""
        
        categories = [
            ('same_instance', same_instance),
            ('intra_cluster', intra_cluster),
            ('inter_cluster', inter_cluster),
            ('random_baseline', random_baseline)
        ]
        
        # Filter out categories with no data
        valid_categories = [(name, stats) for name, stats in categories 
                          if stats['count'] > 0 and stats['mean'] is not None]
        
        # Sort by mean similarity (highest first)
        ranked = sorted(valid_categories, key=lambda x: x[1]['mean'], reverse=True)
        
        return {
            'ranking': [name for name, _ in ranked],
            'values': {name: stats['mean'] for name, stats in ranked},
            'expected_order': ['same_instance', 'intra_cluster', 'inter_cluster', 'random_baseline'],
            'order_correct': [name for name, _ in ranked] == ['same_instance', 'intra_cluster', 'inter_cluster', 'random_baseline']
        }

