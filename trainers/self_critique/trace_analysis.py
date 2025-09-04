"""
Trace Analyzer for TOPAS ARC Solver - Phase 3
Understands why traces work/fail and extracts success patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import networkx as nx

@dataclass
class TraceFeatures:
    """Comprehensive features extracted from a trace"""
    # Program structure
    program_length: int
    unique_operations: int
    operation_frequency: Dict[str, int]
    operation_sequence: List[str]
    max_nesting_depth: int
    
    # Grid transformations
    input_size: Tuple[int, int]
    output_size: Tuple[int, int]
    size_preserving: bool
    color_preserving: bool
    symmetry_preserving: bool
    
    # Execution characteristics
    execution_steps: int
    confidence_score: float
    probability_entropy: float
    
    # Semantic properties
    uses_loops: bool
    uses_conditionals: bool
    uses_recursion: bool
    transformation_type: str
    
    # Success predictors
    grid_coverage: float
    pattern_complexity: float
    logical_consistency: float
    
    # ✅ NEW: Planner features
    planner_halt_confidence: float = 0.0
    planner_entropy: float = 0.0
    planner_alignment: float = 0.0
    
    # ✅ NEW: RelMem features  
    relmem_inherit_loss: float = 0.0
    relmem_inverse_loss: float = 0.0
    relmem_concept_count: int = 0
    
@dataclass
class PatternCluster:
    """Cluster of similar successful patterns"""
    cluster_id: int
    traces: List[Any]
    features: TraceFeatures
    success_rate: float
    common_operations: List[str]
    cluster_center: np.ndarray
    pattern_signature: str
    
@dataclass
class FailureAnalysis:
    """Analysis of why traces fail"""
    failure_type: str
    frequency: int
    common_errors: List[str]
    suggested_fixes: List[str]
    example_traces: List[Any]

class TraceAnalyzer:
    """Analyze traces to understand success patterns and failure modes"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Pattern databases
        self.success_patterns = []
        self.failure_patterns = []
        self.pattern_clusters = []
        
        # Success predictors
        self.success_predictor = None
        self.feature_importance = {}
        
        # Operation semantics
        self.operation_semantics = self._build_operation_semantics()
        
        # Transformation types
        self.transformation_types = {
            'identity': 'No change',
            'color_map': 'Color transformation',
            'spatial': 'Spatial transformation',
            'pattern': 'Pattern manipulation',
            'logical': 'Logical operation',
            'composition': 'Multiple operations'
        }
        
        self.logger = logging.getLogger(__name__)
        
    def analyze_trace(self, trace, task=None, planner_metrics=None, relmem_metrics=None) -> TraceFeatures:
        """Extract comprehensive features from a trace"""
        
        features = TraceFeatures(
            # Program structure
            program_length=len(trace.program) if hasattr(trace, 'program') and trace.program else 0,
            unique_operations=len(set(trace.operations)) if hasattr(trace, 'operations') and trace.operations else 0,
            operation_frequency=Counter(trace.operations) if hasattr(trace, 'operations') and trace.operations else {},
            operation_sequence=trace.operations if hasattr(trace, 'operations') and trace.operations else [],
            max_nesting_depth=self._compute_nesting_depth(trace),
            
            # Grid properties
            input_size=trace.grid_states[0].shape[-2:] if hasattr(trace, 'grid_states') and trace.grid_states else (0, 0),
            output_size=trace.final_grid.shape[-2:] if hasattr(trace, 'final_grid') and trace.final_grid is not None else (0, 0),
            size_preserving=self._check_size_preserving(trace),
            color_preserving=self._check_color_preserving(trace),
            symmetry_preserving=self._check_symmetry_preserving(trace),
            
            # Execution
            execution_steps=len(trace.grid_states) if hasattr(trace, 'grid_states') and trace.grid_states else 0,
            confidence_score=trace.confidence if hasattr(trace, 'confidence') else 0.0,
            probability_entropy=self._compute_probability_entropy(trace),
            
            # Semantic properties
            uses_loops=self._uses_loops(trace),
            uses_conditionals=self._uses_conditionals(trace),
            uses_recursion=self._uses_recursion(trace),
            transformation_type=self._classify_transformation(trace),
            
            # Success predictors
            grid_coverage=self._compute_grid_coverage(trace),
            pattern_complexity=self._compute_pattern_complexity(trace),
            logical_consistency=self._compute_logical_consistency(trace),
            
            # ✅ NEW: Planner features
            planner_halt_confidence=planner_metrics.get('q_halt_mean', 0.0) if planner_metrics else 0.0,
            planner_entropy=planner_metrics.get('entropy', 0.0) if planner_metrics else 0.0,
            planner_alignment=self._compute_planner_trace_alignment(trace, planner_metrics),
            
            # ✅ NEW: RelMem features
            relmem_inherit_loss=relmem_metrics.get('inherit', 0.0) if relmem_metrics else 0.0,
            relmem_inverse_loss=relmem_metrics.get('inverse', 0.0) if relmem_metrics else 0.0,
            relmem_concept_count=self._estimate_concept_count(trace, relmem_metrics)
        )
        
        return features
    
    def learn_success_patterns(self, successful_traces: List, tasks: List = None) -> Dict:
        """Learn what makes traces successful through clustering and analysis"""
        
        if not successful_traces:
            return {'patterns': [], 'insights': []}
        
        self.logger.info(f"Analyzing {len(successful_traces)} successful traces")
        
        # Extract features from all successful traces
        trace_features = []
        feature_vectors = []
        
        for i, trace in enumerate(successful_traces):
            features = self.analyze_trace(trace, tasks[i] if tasks and i < len(tasks) else None)
            trace_features.append(features)
            
            # Convert to feature vector for clustering
            feature_vector = self._features_to_vector(features)
            feature_vectors.append(feature_vector)
        
        if not feature_vectors:
            return {'patterns': [], 'insights': []}
            
        feature_matrix = np.array(feature_vectors)
        
        # Cluster successful patterns
        clusters = self._cluster_success_patterns(feature_matrix, trace_features, successful_traces)
        
        # Analyze each cluster
        pattern_analysis = []
        for cluster in clusters:
            analysis = self._analyze_pattern_cluster(cluster)
            pattern_analysis.append(analysis)
            
        # Extract insights
        insights = self._extract_success_insights(clusters, trace_features)
        
        # Build success predictor
        self._build_success_predictor(trace_features, [1] * len(trace_features))  # All successful
        
        self.success_patterns = clusters
        
        return {
            'patterns': [self._cluster_to_dict(c) for c in clusters],
            'insights': insights,
            'predictor_accuracy': self._validate_success_predictor()
        }
    
    def analyze_failures(self, failed_traces: List, tasks: List = None) -> Dict:
        """Analyze failure patterns and suggest improvements"""
        
        if not failed_traces:
            return {'failures': [], 'suggestions': []}
            
        self.logger.info(f"Analyzing {len(failed_traces)} failed traces")
        
        # Extract features and categorize failures
        failure_categories = defaultdict(list)
        
        for i, trace in enumerate(failed_traces):
            features = self.analyze_trace(trace, tasks[i] if tasks and i < len(tasks) else None)
            failure_type = self._categorize_failure(trace, features)
            failure_categories[failure_type].append((trace, features))
        
        # Analyze each failure category
        failure_analyses = []
        for failure_type, trace_feature_pairs in failure_categories.items():
            analysis = self._analyze_failure_category(failure_type, trace_feature_pairs)
            failure_analyses.append(analysis)
        
        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(failure_analyses)
        
        self.failure_patterns = failure_analyses
        
        return {
            'failures': [self._failure_to_dict(f) for f in failure_analyses],
            'suggestions': suggestions,
            'failure_distribution': {k: len(v) for k, v in failure_categories.items()}
        }
    
    def compare_success_failure(self, successful_traces: List, failed_traces: List) -> Dict:
        """Compare successful and failed traces to find key differences"""
        
        success_features = [self.analyze_trace(trace) for trace in successful_traces]
        failure_features = [self.analyze_trace(trace) for trace in failed_traces]
        
        # Statistical comparison
        comparisons = {}
        
        # Program characteristics
        comparisons['program_length'] = self._compare_distributions(
            [f.program_length for f in success_features],
            [f.program_length for f in failure_features],
            'program_length'
        )
        
        comparisons['unique_operations'] = self._compare_distributions(
            [f.unique_operations for f in success_features],
            [f.unique_operations for f in failure_features],
            'unique_operations'
        )
        
        comparisons['confidence'] = self._compare_distributions(
            [f.confidence_score for f in success_features],
            [f.confidence_score for f in failure_features],
            'confidence_score'
        )
        
        # Operation usage patterns
        success_ops = defaultdict(int)
        failure_ops = defaultdict(int)
        
        for features in success_features:
            for op, count in features.operation_frequency.items():
                success_ops[op] += count
                
        for features in failure_features:
            for op, count in features.operation_frequency.items():
                failure_ops[op] += count
        
        # Operations that correlate with success
        success_correlations = self._find_success_correlations(success_ops, failure_ops)
        
        # Transformation type analysis
        success_transforms = Counter(f.transformation_type for f in success_features)
        failure_transforms = Counter(f.transformation_type for f in failure_features)
        
        return {
            'feature_comparisons': comparisons,
            'operation_correlations': success_correlations,
            'transformation_analysis': {
                'success_distribution': dict(success_transforms),
                'failure_distribution': dict(failure_transforms)
            },
            'key_differences': self._identify_key_differences(success_features, failure_features)
        }
    
    def predict_trace_success(self, trace, task=None) -> Tuple[float, Dict]:
        """Predict if a trace will succeed and provide explanation"""
        
        features = self.analyze_trace(trace, task)
        
        if self.success_predictor is None:
            return 0.5, {'explanation': 'No predictor trained yet'}
        
        # Convert features to prediction input
        feature_vector = self._features_to_vector(features)
        
        try:
            # Simple success prediction (would use trained model in practice)
            success_probability = self._compute_success_probability(features)
            
            explanation = self._explain_prediction(features, success_probability)
            
            return success_probability, explanation
            
        except Exception as e:
            self.logger.warning(f"Prediction failed: {e}")
            return 0.5, {'explanation': f'Prediction error: {e}'}
    
    def _features_to_vector(self, features: TraceFeatures) -> np.ndarray:
        """Convert trace features to numerical vector"""
        
        vector = [
            features.program_length,
            features.unique_operations,
            features.max_nesting_depth,
            features.input_size[0] * features.input_size[1],
            features.output_size[0] * features.output_size[1],
            int(features.size_preserving),
            int(features.color_preserving),
            int(features.symmetry_preserving),
            features.execution_steps,
            features.confidence_score,
            features.probability_entropy,
            int(features.uses_loops),
            int(features.uses_conditionals),
            int(features.uses_recursion),
            features.grid_coverage,
            features.pattern_complexity,
            features.logical_consistency
        ]
        
        return np.array(vector, dtype=np.float32)
    
    def _compute_nesting_depth(self, trace) -> int:
        """Compute maximum nesting depth in program"""
        if not hasattr(trace, 'program') or not trace.program:
            return 0
            
        max_depth = 0
        current_depth = 0
        
        for instruction in trace.program:
            if 'for' in instruction.lower() or 'while' in instruction.lower():
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif 'end' in instruction.lower():
                current_depth = max(0, current_depth - 1)
                
        return max_depth
    
    def _check_size_preserving(self, trace) -> bool:
        """Check if transformation preserves grid size"""
        if not hasattr(trace, 'grid_states') or not trace.grid_states:
            return True
            
        input_size = trace.grid_states[0].shape[-2:]
        output_size = trace.final_grid.shape[-2:] if hasattr(trace, 'final_grid') and trace.final_grid is not None else input_size
        
        return input_size == output_size
    
    def _check_color_preserving(self, trace) -> bool:
        """Check if transformation preserves color set"""
        if not hasattr(trace, 'grid_states') or not trace.grid_states or not hasattr(trace, 'final_grid'):
            return True
            
        input_colors = set(torch.unique(trace.grid_states[0]).tolist())
        output_colors = set(torch.unique(trace.final_grid).tolist()) if trace.final_grid is not None else input_colors
        
        return input_colors == output_colors
    
    def _check_symmetry_preserving(self, trace) -> bool:
        """Check if transformation preserves symmetries"""
        if not hasattr(trace, 'final_grid') or trace.final_grid is None:
            return True
            
        grid = trace.final_grid
        
        # Check horizontal symmetry
        h_symmetric = torch.equal(grid, torch.flip(grid, dims=[-1]))
        
        # Check vertical symmetry  
        v_symmetric = torch.equal(grid, torch.flip(grid, dims=[-2]))
        
        return h_symmetric or v_symmetric
    
    def _compute_probability_entropy(self, trace) -> float:
        """Compute entropy of probability distribution"""
        if not hasattr(trace, 'probabilities') or trace.probabilities is None or len(trace.probabilities) == 0:
            return 0.0
            
        probs = F.softmax(trace.probabilities, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        
        return entropy
    
    def _uses_loops(self, trace) -> bool:
        """Check if trace uses loop constructs"""
        if not hasattr(trace, 'operations') or not trace.operations:
            return False
            
        loop_keywords = ['for', 'while', 'repeat', 'iterate']
        return any(any(kw in op.lower() for kw in loop_keywords) for op in trace.operations)
    
    def _uses_conditionals(self, trace) -> bool:
        """Check if trace uses conditional logic"""
        if not hasattr(trace, 'operations') or not trace.operations:
            return False
            
        conditional_keywords = ['if', 'when', 'unless', 'case']
        return any(any(kw in op.lower() for kw in conditional_keywords) for op in trace.operations)
    
    def _uses_recursion(self, trace) -> bool:
        """Check if trace uses recursive operations"""
        if not hasattr(trace, 'operations') or not trace.operations:
            return False
            
        recursive_keywords = ['recursive', 'self', 'recurse']
        return any(any(kw in op.lower() for kw in recursive_keywords) for op in trace.operations)
    
    def _classify_transformation(self, trace) -> str:
        """Classify the type of transformation"""
        if not hasattr(trace, 'operations') or not trace.operations:
            return 'identity'
            
        ops = [op.lower() for op in trace.operations]
        
        # Color operations
        if any('color' in op or 'fill' in op or 'paint' in op for op in ops):
            return 'color_map'
            
        # Spatial operations
        if any('rotate' in op or 'flip' in op or 'mirror' in op or 'translate' in op for op in ops):
            return 'spatial'
            
        # Pattern operations
        if any('pattern' in op or 'repeat' in op or 'tile' in op for op in ops):
            return 'pattern'
            
        # Logical operations
        if any('and' in op or 'or' in op or 'not' in op or 'xor' in op for op in ops):
            return 'logical'
            
        # Multiple operation types
        if len(set(ops)) > 3:
            return 'composition'
            
        return 'identity'
    
    def _compute_grid_coverage(self, trace) -> float:
        """Compute how much of the grid is affected"""
        if not hasattr(trace, 'final_grid') or trace.final_grid is None:
            return 0.0
            
        if not hasattr(trace, 'grid_states') or not trace.grid_states:
            return 0.0
            
        initial = trace.grid_states[0]
        final = trace.final_grid
        
        if initial.shape != final.shape:
            return 1.0  # Full coverage if size changed
            
        changed_pixels = (initial != final).sum().item()
        total_pixels = initial.numel()
        
        return changed_pixels / total_pixels if total_pixels > 0 else 0.0
    
    def _compute_pattern_complexity(self, trace) -> float:
        """Compute complexity of pattern in final grid"""
        if not hasattr(trace, 'final_grid') or trace.final_grid is None:
            return 0.0
            
        grid = trace.final_grid
        
        # Color diversity
        unique_colors = len(torch.unique(grid))
        color_complexity = unique_colors / 10.0  # Normalize by max colors
        
        # Spatial complexity (edges)
        def count_edges(grid):
            edges = 0
            h, w = grid.shape[-2:]
            for i in range(h-1):
                for j in range(w-1):
                    if grid[i, j] != grid[i+1, j]:
                        edges += 1
                    if grid[i, j] != grid[i, j+1]:
                        edges += 1
            return edges
            
        total_edges = count_edges(grid)
        max_edges = 2 * (grid.shape[-1] - 1) * (grid.shape[-2] - 1)
        spatial_complexity = total_edges / max(max_edges, 1)
        
        return 0.5 * color_complexity + 0.5 * spatial_complexity
    
    def _compute_logical_consistency(self, trace) -> float:
        """Compute logical consistency of the trace"""
        if not hasattr(trace, 'reasoning_steps') or not trace.reasoning_steps:
            return 1.0  # Assume consistent if no reasoning provided
            
        # Simple consistency check - look for contradictions
        steps_text = ' '.join(trace.reasoning_steps).lower()
        
        contradictions = [
            ('increase', 'decrease'),
            ('add', 'remove'),
            ('same', 'different'),
            ('connect', 'separate')
        ]
        
        inconsistencies = 0
        for pos, neg in contradictions:
            if pos in steps_text and neg in steps_text:
                inconsistencies += 1
                
        return max(0.0, 1.0 - inconsistencies * 0.2)
    
    def _cluster_success_patterns(self, feature_matrix: np.ndarray, trace_features: List[TraceFeatures], traces: List) -> List[PatternCluster]:
        """Cluster successful patterns using K-means"""
        
        if len(feature_matrix) < 3:
            # Too few samples for clustering
            return [PatternCluster(
                cluster_id=0,
                traces=traces,
                features=trace_features[0] if trace_features else TraceFeatures(),
                success_rate=1.0,
                common_operations=[],
                cluster_center=feature_matrix[0] if len(feature_matrix) > 0 else np.array([]),
                pattern_signature="single_pattern"
            )]
        
        # Determine optimal number of clusters
        max_clusters = min(5, len(feature_matrix) // 2)
        best_score = -1
        best_k = 2
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(feature_matrix)
            
            if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette score
                score = silhouette_score(feature_matrix, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        
        # Final clustering
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feature_matrix)
        
        clusters = []
        for cluster_id in range(best_k):
            cluster_mask = labels == cluster_id
            cluster_traces = [traces[i] for i in range(len(traces)) if cluster_mask[i]]
            cluster_features = [trace_features[i] for i in range(len(trace_features)) if cluster_mask[i]]
            
            if cluster_traces:
                # Analyze cluster characteristics
                common_ops = self._find_common_operations(cluster_features)
                pattern_sig = self._generate_pattern_signature(cluster_features)
                
                cluster = PatternCluster(
                    cluster_id=cluster_id,
                    traces=cluster_traces,
                    features=cluster_features[0],  # Representative features
                    success_rate=1.0,  # All are successful
                    common_operations=common_ops,
                    cluster_center=kmeans.cluster_centers_[cluster_id],
                    pattern_signature=pattern_sig
                )
                clusters.append(cluster)
        
        return clusters
    
    def _find_common_operations(self, features_list: List[TraceFeatures]) -> List[str]:
        """Find operations common to most traces in cluster"""
        
        all_ops = []
        for features in features_list:
            all_ops.extend(features.operation_sequence)
        
        op_counts = Counter(all_ops)
        threshold = len(features_list) * 0.5  # Present in at least 50% of traces
        
        common_ops = [op for op, count in op_counts.items() if count >= threshold]
        return common_ops
    
    def _generate_pattern_signature(self, features_list: List[TraceFeatures]) -> str:
        """Generate signature describing the pattern"""
        
        if not features_list:
            return "empty"
            
        # Analyze common characteristics
        avg_length = np.mean([f.program_length for f in features_list])
        common_transform = Counter([f.transformation_type for f in features_list]).most_common(1)[0][0]
        uses_loops = sum(f.uses_loops for f in features_list) / len(features_list)
        
        signature_parts = [
            f"{common_transform}",
            f"len_{int(avg_length)}",
        ]
        
        if uses_loops > 0.5:
            signature_parts.append("loops")
            
        return "_".join(signature_parts)
    
    def _analyze_pattern_cluster(self, cluster: PatternCluster) -> Dict:
        """Analyze a cluster of successful patterns"""
        
        analysis = {
            'cluster_id': cluster.cluster_id,
            'size': len(cluster.traces),
            'common_operations': cluster.common_operations,
            'transformation_type': cluster.features.transformation_type,
            'avg_confidence': np.mean([t.confidence for t in cluster.traces if hasattr(t, 'confidence')]),
            'pattern_signature': cluster.pattern_signature,
            'key_characteristics': []
        }
        
        # Identify key characteristics
        if cluster.features.uses_loops:
            analysis['key_characteristics'].append('uses_iterative_operations')
            
        if cluster.features.size_preserving:
            analysis['key_characteristics'].append('preserves_grid_size')
            
        if cluster.features.symmetry_preserving:
            analysis['key_characteristics'].append('maintains_symmetry')
            
        if cluster.features.pattern_complexity > 0.7:
            analysis['key_characteristics'].append('high_pattern_complexity')
            
        return analysis
    
    def _extract_success_insights(self, clusters: List[PatternCluster], all_features: List[TraceFeatures]) -> List[str]:
        """Extract insights about what makes traces successful"""
        
        insights = []
        
        # Cluster-based insights
        if len(clusters) > 1:
            insights.append(f"Success patterns fall into {len(clusters)} distinct categories")
            
        # Operation insights
        all_common_ops = set()
        for cluster in clusters:
            all_common_ops.update(cluster.common_operations)
            
        if all_common_ops:
            insights.append(f"Key operations for success: {', '.join(list(all_common_ops)[:5])}")
        
        # Transformation insights
        transform_distribution = Counter([f.transformation_type for f in all_features])
        most_successful_transform = transform_distribution.most_common(1)[0]
        insights.append(f"Most successful transformation type: {most_successful_transform[0]} ({most_successful_transform[1]} instances)")
        
        # Complexity insights
        avg_complexity = np.mean([f.pattern_complexity for f in all_features])
        if avg_complexity > 0.6:
            insights.append("Successful traces tend to create complex patterns")
        elif avg_complexity < 0.3:
            insights.append("Successful traces tend to create simple patterns")
        
        # Consistency insights
        avg_consistency = np.mean([f.logical_consistency for f in all_features])
        if avg_consistency > 0.8:
            insights.append("High logical consistency is crucial for success")
        
        return insights
    
    def _categorize_failure(self, trace, features: TraceFeatures) -> str:
        """Categorize the type of failure"""
        
        if features.confidence_score < 0.3:
            return "low_confidence"
        elif features.program_length == 0:
            return "no_program"
        elif features.logical_consistency < 0.5:
            return "logical_inconsistency"
        elif features.grid_coverage < 0.1:
            return "insufficient_transformation"
        elif not features.size_preserving and features.transformation_type != 'spatial':
            return "unexpected_size_change"
        else:
            return "unknown_failure"
    
    def _analyze_failure_category(self, failure_type: str, trace_feature_pairs: List) -> FailureAnalysis:
        """Analyze a specific category of failures"""
        
        traces = [pair[0] for pair in trace_feature_pairs]
        features = [pair[1] for pair in trace_feature_pairs]
        
        # Common errors in this category
        common_errors = []
        if failure_type == "low_confidence":
            common_errors = ["Model uncertainty", "Ambiguous input", "Insufficient training"]
        elif failure_type == "logical_inconsistency":
            common_errors = ["Contradictory operations", "Invalid sequence", "Missing constraints"]
        elif failure_type == "insufficient_transformation":
            common_errors = ["Incomplete operation", "Wrong target selection", "Premature termination"]
        
        # Generate suggestions
        suggestions = []
        if failure_type == "low_confidence":
            suggestions = ["Increase training data", "Improve model architecture", "Add confidence regularization"]
        elif failure_type == "logical_inconsistency":
            suggestions = ["Add logical consistency checks", "Improve reasoning module", "Better program validation"]
        elif failure_type == "insufficient_transformation":
            suggestions = ["Extend operation sequences", "Improve target detection", "Add completion checks"]
        
        return FailureAnalysis(
            failure_type=failure_type,
            frequency=len(traces),
            common_errors=common_errors,
            suggested_fixes=suggestions,
            example_traces=traces[:3]  # Keep a few examples
        )
    
    def _generate_improvement_suggestions(self, failure_analyses: List[FailureAnalysis]) -> List[str]:
        """Generate overall improvement suggestions"""
        
        suggestions = []
        
        # Most common failure type
        most_common = max(failure_analyses, key=lambda x: x.frequency)
        suggestions.append(f"Priority fix: Address {most_common.failure_type} ({most_common.frequency} instances)")
        
        # Collect all suggested fixes
        all_fixes = []
        for analysis in failure_analyses:
            all_fixes.extend(analysis.suggested_fixes)
        
        # Most common suggestions
        fix_counts = Counter(all_fixes)
        for fix, count in fix_counts.most_common(3):
            suggestions.append(f"Recommended: {fix} (addresses {count} failure types)")
        
        return suggestions
    
    def _compare_distributions(self, success_values: List, failure_values: List, feature_name: str) -> Dict:
        """Compare distributions between success and failure"""
        
        if not success_values or not failure_values:
            return {'difference': 'insufficient_data'}
        
        success_mean = np.mean(success_values)
        failure_mean = np.mean(failure_values)
        
        success_std = np.std(success_values)
        failure_std = np.std(failure_values)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((success_std**2 + failure_std**2) / 2)
        effect_size = (success_mean - failure_mean) / pooled_std if pooled_std > 0 else 0
        
        return {
            'success_mean': success_mean,
            'failure_mean': failure_mean,
            'difference': success_mean - failure_mean,
            'effect_size': effect_size,
            'significance': 'large' if abs(effect_size) > 0.8 else 'medium' if abs(effect_size) > 0.5 else 'small'
        }
    
    def _find_success_correlations(self, success_ops: Dict, failure_ops: Dict) -> Dict:
        """Find operations that correlate with success"""
        
        correlations = {}
        
        total_success = sum(success_ops.values())
        total_failure = sum(failure_ops.values())
        
        for op in set(success_ops.keys()) | set(failure_ops.keys()):
            success_rate = success_ops.get(op, 0) / max(total_success, 1)
            failure_rate = failure_ops.get(op, 0) / max(total_failure, 1)
            
            correlation = success_rate - failure_rate
            correlations[op] = {
                'correlation': correlation,
                'success_rate': success_rate,
                'failure_rate': failure_rate
            }
        
        # Sort by correlation strength
        sorted_ops = sorted(correlations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
        
        return dict(sorted_ops[:10])  # Top 10
    
    def _identify_key_differences(self, success_features: List[TraceFeatures], failure_features: List[TraceFeatures]) -> List[str]:
        """Identify key differences between success and failure"""
        
        differences = []
        
        # Confidence difference
        success_conf = np.mean([f.confidence_score for f in success_features])
        failure_conf = np.mean([f.confidence_score for f in failure_features])
        
        if success_conf - failure_conf > 0.2:
            differences.append(f"Successful traces have higher confidence ({success_conf:.2f} vs {failure_conf:.2f})")
        
        # Program length difference
        success_len = np.mean([f.program_length for f in success_features])
        failure_len = np.mean([f.program_length for f in failure_features])
        
        if abs(success_len - failure_len) > 2:
            if success_len > failure_len:
                differences.append(f"Successful traces use longer programs ({success_len:.1f} vs {failure_len:.1f} operations)")
            else:
                differences.append(f"Successful traces use shorter programs ({success_len:.1f} vs {failure_len:.1f} operations)")
        
        # Logical consistency
        success_logic = np.mean([f.logical_consistency for f in success_features])
        failure_logic = np.mean([f.logical_consistency for f in failure_features])
        
        if success_logic - failure_logic > 0.2:
            differences.append(f"Successful traces are more logically consistent ({success_logic:.2f} vs {failure_logic:.2f})")
        
        return differences
    
    def _compute_success_probability(self, features: TraceFeatures) -> float:
        """Simple success probability based on features"""
        
        # Weighted combination of key features
        prob = 0.0
        
        # Confidence is important
        prob += 0.3 * features.confidence_score
        
        # Logical consistency
        prob += 0.25 * features.logical_consistency
        
        # Reasonable program length (not too short or long)
        optimal_length = 10
        length_score = 1.0 - abs(features.program_length - optimal_length) / optimal_length
        prob += 0.2 * max(0, length_score)
        
        # Grid coverage (but not too much)
        coverage_score = features.grid_coverage if features.grid_coverage < 0.8 else 1.0 - features.grid_coverage
        prob += 0.15 * coverage_score
        
        # Pattern complexity (moderate is good)
        complexity_score = 1.0 - abs(features.pattern_complexity - 0.5) * 2
        prob += 0.1 * max(0, complexity_score)
        
        return max(0.0, min(1.0, prob))
    
    def _explain_prediction(self, features: TraceFeatures, probability: float) -> Dict:
        """Explain success prediction"""
        
        explanation = {
            'probability': probability,
            'factors': [],
            'recommendations': []
        }
        
        # Analyze individual factors
        if features.confidence_score > 0.7:
            explanation['factors'].append("High model confidence")
        elif features.confidence_score < 0.3:
            explanation['factors'].append("Low model confidence (concerning)")
            explanation['recommendations'].append("Increase model confidence through better training")
        
        if features.logical_consistency > 0.8:
            explanation['factors'].append("High logical consistency")
        elif features.logical_consistency < 0.5:
            explanation['factors'].append("Poor logical consistency (major issue)")
            explanation['recommendations'].append("Improve reasoning logic and consistency checks")
        
        if 5 <= features.program_length <= 15:
            explanation['factors'].append("Optimal program length")
        elif features.program_length < 3:
            explanation['factors'].append("Program too short (may be incomplete)")
            explanation['recommendations'].append("Extend program with additional operations")
        elif features.program_length > 20:
            explanation['factors'].append("Program too long (may be inefficient)")
            explanation['recommendations'].append("Simplify program by removing redundant operations")
        
        return explanation
    
    def _build_success_predictor(self, features_list: List[TraceFeatures], labels: List[int]):
        """Build a simple success predictor"""
        # In practice, would train a more sophisticated model
        # For now, just store feature importance
        
        if len(features_list) < 5:
            return
            
        # Compute feature importance based on correlation with success
        feature_vectors = [self._features_to_vector(f) for f in features_list]
        feature_matrix = np.array(feature_vectors)
        
        # Simple correlation analysis
        feature_names = [
            'program_length', 'unique_operations', 'max_nesting_depth',
            'grid_size', 'output_size', 'size_preserving', 'color_preserving',
            'symmetry_preserving', 'execution_steps', 'confidence_score',
            'probability_entropy', 'uses_loops', 'uses_conditionals',
            'uses_recursion', 'grid_coverage', 'pattern_complexity',
            'logical_consistency'
        ]
        
        for i, name in enumerate(feature_names):
            if i < feature_matrix.shape[1]:
                correlation = np.corrcoef(feature_matrix[:, i], labels)[0, 1]
                self.feature_importance[name] = correlation if not np.isnan(correlation) else 0.0
        
        self.success_predictor = True  # Mark as built
    
    def _validate_success_predictor(self) -> float:
        """Validate success predictor accuracy"""
        # Placeholder - would use cross-validation in practice
        return 0.75 if self.success_predictor else 0.5
    
    def _cluster_to_dict(self, cluster: PatternCluster) -> Dict:
        """Convert cluster to dictionary for serialization"""
        return {
            'cluster_id': cluster.cluster_id,
            'size': len(cluster.traces),
            'success_rate': cluster.success_rate,
            'common_operations': cluster.common_operations,
            'pattern_signature': cluster.pattern_signature
        }
    
    def _failure_to_dict(self, failure: FailureAnalysis) -> Dict:
        """Convert failure analysis to dictionary"""
        return {
            'failure_type': failure.failure_type,
            'frequency': failure.frequency,
            'common_errors': failure.common_errors,
            'suggested_fixes': failure.suggested_fixes,
            'example_count': len(failure.example_traces)
        }
    
    def _build_operation_semantics(self) -> Dict:
        """Build semantic understanding of operations"""
        return {
            'spatial': ['rotate', 'flip', 'mirror', 'translate', 'scale'],
            'color': ['color_map', 'fill', 'replace', 'invert'],
            'logical': ['and', 'or', 'not', 'xor'],
            'structural': ['extract', 'overlay', 'connect', 'separate'],
            'pattern': ['repeat', 'tile', 'extend', 'shrink']
        }
    
    # ✅ NEW: Helper methods for planner and RelMem features
    def _compute_planner_trace_alignment(self, trace, planner_metrics) -> float:
        """Compute how well trace aligns with planner expectations"""
        
        if not planner_metrics or not hasattr(trace, 'operations'):
            return 0.0
        
        # Simple heuristic: higher entropy = more uncertainty = lower alignment
        entropy = planner_metrics.get('entropy', 0.0)
        halt_confidence = planner_metrics.get('q_halt_mean', 0.5)
        
        # High halt confidence + low entropy = good alignment
        if halt_confidence > 0.7 and entropy < 0.5:
            return 0.9
        elif halt_confidence < 0.3 or entropy > 1.0:
            return 0.1
        else:
            # Interpolate based on confidence and entropy
            conf_score = halt_confidence
            entropy_score = max(0, 1 - entropy)
            return (conf_score + entropy_score) / 2
    
    def _estimate_concept_count(self, trace, relmem_metrics) -> int:
        """Estimate number of relational concepts used in trace"""
        
        if not relmem_metrics:
            return 0
            
        # Heuristic: lower inherit/inverse losses suggest more stable concepts
        inherit_loss = relmem_metrics.get('inherit', 1.0)
        inverse_loss = relmem_metrics.get('inverse', 1.0)
        
        # More stable relations = more concepts being used effectively
        stability = max(0, 2 - inherit_loss - inverse_loss)
        
        # Estimate concept count based on trace complexity and stability
        if hasattr(trace, 'operations') and trace.operations:
            base_concepts = len(set(trace.operations))  # Unique operations
            concept_multiplier = min(3.0, 1 + stability)  # Stability boost
            return int(base_concepts * concept_multiplier)
        
        return max(1, int(stability * 2))  # Minimum fallback