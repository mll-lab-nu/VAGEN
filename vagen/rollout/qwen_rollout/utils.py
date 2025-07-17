import torch

def compute_loss_mask(input_ids, attention_mask, sptk_b, sptk_e, pad_token_id):
    """
    Compute loss mask for tokens wrapped by special tokens and remove special tokens.
    
    Args:
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        sptk_b: start special token id
        sptk_e: end special token id
        pad_token_id: padding token id
    
    Returns:
        input_ids: (batch_size, seq_len) - with special tokens removed
        attention_mask: (batch_size, seq_len) - updated accordingly
        loss_mask: (batch_size, seq_len) - 1 for tokens to compute loss on
        end_of_response_position_mask: (batch_size, seq_len) - 1 at last token of each response
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    # Initialize output tensors
    new_input_ids = torch.full_like(input_ids, pad_token_id)
    new_attention_mask = torch.zeros_like(attention_mask)
    loss_mask = torch.zeros_like(attention_mask)
    end_of_response_position_mask = torch.zeros_like(attention_mask)
    
    for b in range(batch_size):
        # Find special token positions
        sptk_b_mask = (input_ids[b] == sptk_b)
        sptk_e_mask = (input_ids[b] == sptk_e)
        
        # Get indices of special tokens
        sptk_b_indices = sptk_b_mask.nonzero(as_tuple=True)[0]
        sptk_e_indices = sptk_e_mask.nonzero(as_tuple=True)[0]
        
        # Handle edge cases
        # Case 1: Truncated start - first sptk_e appears before first sptk_b
        if len(sptk_e_indices) > 0 and (len(sptk_b_indices) == 0 or sptk_e_indices[0] < sptk_b_indices[0]):
            # Add a virtual sptk_b at position -1
            sptk_b_indices = torch.cat([torch.tensor([-1], device=device), sptk_b_indices])
        
        # Case 2: Truncated end - last sptk_b has no corresponding sptk_e
        if len(sptk_b_indices) > len(sptk_e_indices):
            # Add a virtual sptk_e at the last non-pad position
            last_non_pad = (input_ids[b] != pad_token_id).nonzero(as_tuple=True)[0]
            if len(last_non_pad) > 0:
                virtual_end = last_non_pad[-1] + 1
            else:
                virtual_end = seq_len
            sptk_e_indices = torch.cat([sptk_e_indices, torch.tensor([virtual_end], device=device)])
        
        # Create mask for all special tokens to remove
        special_token_mask = sptk_b_mask | sptk_e_mask
        
        # Create loss mask for tokens between special tokens
        temp_loss_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        temp_end_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        
        for start_idx, end_idx in zip(sptk_b_indices, sptk_e_indices):
            if start_idx >= 0 and end_idx < seq_len:
                # Normal case: both tokens are within bounds
                temp_loss_mask[start_idx + 1:end_idx] = True
                if end_idx > start_idx + 1:  # Ensure there's at least one token between
                    temp_end_mask[end_idx - 1] = True
            elif start_idx < 0 and end_idx < seq_len:
                # Truncated start case
                temp_loss_mask[0:end_idx] = True
                if end_idx > 0:
                    temp_end_mask[end_idx - 1] = True
            elif start_idx >= 0 and end_idx >= seq_len:
                # Truncated end case
                temp_loss_mask[start_idx + 1:] = True
                # Find last non-pad token in this range
                last_valid = (input_ids[b, start_idx + 1:] != pad_token_id).nonzero(as_tuple=True)[0]
                if len(last_valid) > 0:
                    temp_end_mask[start_idx + 1 + last_valid[-1]] = True
        
        # Get indices of tokens to keep (non-special tokens)
        keep_mask = ~special_token_mask
        keep_indices = keep_mask.nonzero(as_tuple=True)[0]
        
        # Copy non-special tokens to new tensors
        if len(keep_indices) > 0:
            new_input_ids[b, :len(keep_indices)] = input_ids[b, keep_indices]
            new_attention_mask[b, :len(keep_indices)] = attention_mask[b, keep_indices]
            
            # Map loss masks to new positions
            loss_mask[b, :len(keep_indices)] = temp_loss_mask[keep_indices]
            end_of_response_position_mask[b, :len(keep_indices)] = temp_end_mask[keep_indices]
    
    return new_input_ids, new_attention_mask, loss_mask, end_of_response_position_mask


# Example usage and test cases
if __name__ == "__main__":
    # Test case 1: Normal case
    input_ids = torch.tensor([[1, 100, 2, 3, 4, 101, 5, 100, 6, 7, 101, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    sptk_b, sptk_e, pad_token_id = 100, 101, 0
    
    new_ids, new_mask, loss_mask, end_mask = compute_loss_mask(
        input_ids, attention_mask, sptk_b, sptk_e, pad_token_id
    )
    
    print("Test 1 - Normal case:")
    print("Original:", input_ids)
    print("New IDs:", new_ids)
    print("Loss mask:", loss_mask)
    print("End mask:", end_mask)
    
    # Test case 2: Truncated start
    input_ids = torch.tensor([[2, 3, 4, 101, 5, 100, 6, 7, 101, 0, 0, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
    
    new_ids, new_mask, loss_mask, end_mask = compute_loss_mask(
        input_ids, attention_mask, sptk_b, sptk_e, pad_token_id
    )
    
    print("\nTest 2 - Truncated start:")
    print("Original:", input_ids)
    print("New IDs:", new_ids)
    print("Loss mask:", loss_mask)
    print("End mask:", end_mask)
    
    # Test case 3: Truncated end (no padding)
    input_ids = torch.tensor([[1, 100, 2, 3, 4, 101, 5, 100, 6, 7, 8, 9, 10]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    
    new_ids, new_mask, loss_mask, end_mask = compute_loss_mask(
        input_ids, attention_mask, sptk_b, sptk_e, pad_token_id
    )
    print("\nTest 3 - Truncated end:")
    print("Original:", input_ids)
    print("New IDs:", new_ids)
    print("Loss mask:", loss_mask)
    print("End mask:", end_mask)
    # Test case 4: End with sptk_
    input_ids = torch.tensor([[1, 100, 2, 3, 4, 101, 5, 100, 6, 7, 8, 9, 101]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    
    new_ids, new_mask, loss_mask, end_mask = compute_loss_mask(
        input_ids, attention_mask, sptk_b, sptk_e, pad_token_id
    )
    
    print("\nTest 4 - End with sptk_e:")
    print("Original:", input_ids)
    print("New IDs:", new_ids)
    print("Loss mask:", loss_mask)
    print("End mask:", end_mask)
    
    