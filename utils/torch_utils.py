import torch

def cosine_similarity_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    '''
    Compute the pair-wise cosine similarity between two batches of vectors.

    Args:
        a (torch.Tensor, shape (M, D)): The first batch of vectors.
        b (torch.Tensor, shape (N, D)): The second batch of vectors.

    Returns:
        torch.Tensor (shape (M, N)): The pair-wise cosine similarity between the two batches of vectors.
    '''
    
    # Normalize the vectors in batch a and b
    a_norm = a / (a.norm(dim=1, keepdim=True) + 1e-8)  # Adding a small value to prevent division by zero
    b_norm = b / (b.norm(dim=1, keepdim=True) + 1e-8)
    
    # Compute pairwise cosine similarity using matrix multiplication
    sim = torch.mm(a_norm, b_norm.t())
    
    return sim