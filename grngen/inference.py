from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt


def evaluate_inference(true_adj, inferred_adj, gene_names=None, plot=False):
    """
    Evaluate inferred network against ground truth using AUROC and AUPRC.
    
    Parameters
    ----------
    true_adj : np.ndarray or sparse matrix
        Ground truth adjacency matrix (can be signed: -1, 0, +1).
        Any non-zero value is treated as an edge.
    inferred_adj : np.ndarray or sparse matrix
        Inferred adjacency matrix with edge weights/scores.
        Higher values = higher confidence of edge existence.
    gene_names : list, optional
        List of gene names (not used in computation, for future extensions).
    plot : bool, optional
        If True, plot ROC and PR curves.
    
    Returns
    -------
    auroc : float
        Area Under the ROC Curve.
    auprc : float
        Area Under the Precision-Recall Curve.
    """
    # Convert sparse to dense if needed
    if sparse.issparse(true_adj):
        true_adj = true_adj.toarray()
    if sparse.issparse(inferred_adj):
        inferred_adj = inferred_adj.toarray()
    
    # Flatten matrices
    y_true = true_adj.flatten()
    y_scores = inferred_adj.flatten()
    
    # Convert to binary (any edge = 1, no edge = 0)
    # Handles signed matrices: -1 and +1 both become 1
    y_true_binary = (y_true != 0).astype(int)
    
    # Remove self-loops (diagonal) from evaluation
    n = true_adj.shape[0]
    mask = ~np.eye(n, dtype=bool).flatten()
    y_true_binary = y_true_binary[mask]
    y_scores = y_scores[mask]
    
    # Handle edge cases
    if len(np.unique(y_true_binary)) < 2:
        print("Warning: Only one class present in ground truth. Metrics may be undefined.")
        return np.nan, np.nan
    
    # Compute metrics
    auroc = roc_auc_score(y_true_binary, y_scores)
    auprc = average_precision_score(y_true_binary, y_scores)
    
    # Compute baseline (random) AUPRC
    baseline_auprc = y_true_binary.sum() / len(y_true_binary)
    
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
        axes[0].plot(fpr, tpr, 'b-', label=f'AUROC = {auroc:.3f}')
        axes[0].plot([0, 1], [0, 1], 'k--', label='Random (0.500)')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend(loc='lower right')
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1])
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
        axes[1].plot(recall, precision, 'b-', label=f'AUPRC = {auprc:.3f}')
        axes[1].axhline(y=baseline_auprc, color='k', linestyle='--', label=f'Random = {baseline_auprc:.3f}')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].legend(loc='upper right')
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.show()
    
    return auroc, auprc


def evaluate_inference_v2(true_adj, inferred_adj, gene_names=None, plot=False):
    """
    Evaluate inferred network against ground truth using AUROC and AUPRC.
    Optimized to work with sparse matrices without full dense conversion.
    """
    from scipy import sparse
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    n = true_adj.shape[0]
    
    # Ensure sparse format (CSR is efficient for row operations)
    if sparse.issparse(true_adj):
        true_adj = sparse.csr_matrix(true_adj)
    else:
        true_adj = sparse.csr_matrix(true_adj)
    
    if sparse.issparse(inferred_adj):
        inferred_adj = sparse.csr_matrix(inferred_adj)
    else:
        inferred_adj = sparse.csr_matrix(inferred_adj)
    
    # Remove diagonal (self-loops) efficiently
    true_adj = true_adj.copy()
    inferred_adj = inferred_adj.copy()
    true_adj.setdiag(0)
    inferred_adj.setdiag(0)
    true_adj.eliminate_zeros()
    inferred_adj.eliminate_zeros()
    
    # Total number of possible edges (excluding diagonal)
    n_possible_edges = n * n - n
    
    # Get non-zero positions from BOTH matrices
    true_edges = set(zip(*true_adj.nonzero()))
    inferred_nonzero = set(zip(*inferred_adj.nonzero()))
    
    # All positions we need to evaluate (union of both)
    all_relevant_positions = true_edges | inferred_nonzero
    
    if len(all_relevant_positions) == 0:
        print("Warning: No edges in either matrix.")
        return np.nan, np.nan
    
    # Number of true negatives not in our relevant positions
    n_true_negatives_outside = n_possible_edges - len(all_relevant_positions)
    n_true_positives = len(true_edges)
    n_actual_negatives_in_relevant = len(all_relevant_positions) - len(true_edges & all_relevant_positions)
    
    # Build arrays only for relevant positions
    rows, cols = zip(*all_relevant_positions) if all_relevant_positions else ([], [])
    rows, cols = np.array(rows), np.array(cols)
    
    y_true_binary = np.array([(r, c) in true_edges for r, c in zip(rows, cols)], dtype=int)
    y_scores = np.array(inferred_adj[rows, cols]).flatten()
    
    # For positions with zero inferred score that are true negatives,
    # we need to account for them (they all have score 0)
    # Add representative sample of true negatives with score 0
    if n_true_negatives_outside > 0:
        # Add the true negatives (score=0, label=0)
        y_true_binary = np.concatenate([y_true_binary, np.zeros(n_true_negatives_outside, dtype=int)])
        y_scores = np.concatenate([y_scores, np.zeros(n_true_negatives_outside)])
    
    # Handle edge cases
    if len(np.unique(y_true_binary)) < 2:
        print("Warning: Only one class present in ground truth.")
        return np.nan, np.nan
    
    # Compute metrics
    auroc = roc_auc_score(y_true_binary, y_scores)
    auprc = average_precision_score(y_true_binary, y_scores)
    
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
        axes[0].plot(fpr, tpr, 'b-', label=f'AUROC = {auroc:.3f}')
        axes[0].plot([0, 1], [0, 1], 'k--', label='Random (0.500)')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend(loc='lower right')
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1])
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
        axes[1].plot(recall, precision, 'b-', label=f'AUPRC = {auprc:.3f}')
        axes[1].axhline(y=baseline_auprc, color='k', linestyle='--', label=f'Random = {baseline_auprc:.3f}')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].legend(loc='upper right')
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.show()
    
    return auroc, auprc