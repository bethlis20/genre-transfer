import torch
from torch import nn

# Adversarial loss function
def adv_loss(tensor, real=True):
    """
    Adversarial loss for GANs.
    Encourages the discriminator to correctly classify real vs fake data.
    
    Args:
        tensor (torch.Tensor): Output from the discriminator.
        real (bool): If True, calculates loss assuming data is real; otherwise fake.
    
    Returns:
        torch.Tensor: Loss value.
    """
    return torch.mean(torch.clamp(1 - tensor, min=0)) if real else torch.mean(torch.clamp(1 + tensor, min=0))

# Embedding loss function
def embed_loss(a1, a2, ab1, ab2):
    """
    Embedding loss to maintain relative distances in latent space.
    
    Args:
        a1, a2, ab1, ab2 (torch.Tensor): Embeddings for input and transformed data.
    
    Returns:
        torch.Tensor: Loss value.
    """
    mse_loss = nn.MSELoss()
    return torch.mean(((a1 - a2) - (ab1 - ab2)) ** 2) - mse_loss(a1, a2) * mse_loss(ab1, ab2)

# Margin loss function
def margin_loss(a1, a2, delta=2.0):
    """
    Margin loss to enforce minimum separation between embeddings.
    
    Args:
        a1, a2 (torch.Tensor): Embeddings for comparison.
        delta (float): Margin value to enforce.
    
    Returns:
        torch.Tensor: Loss value.
    """
    logits = torch.sqrt(torch.sum((a1 - a2) ** 2, dim=1))
    return torch.mean(torch.clamp(delta - logits, min=0))

# Combined generator loss
def generator_loss(discriminator, real_data, fake_data, encoder, margin=2.0):
    """
    Combined loss function for the generator.
    Includes adversarial loss, embedding loss, and margin loss.
    
    Args:
        discriminator (torch.nn.Module): The discriminator model.
        real_data (torch.Tensor): Real input data.
        fake_data (torch.Tensor): Generated (fake) data.
        encoder (torch.nn.Module): Encoder to compute embeddings.
        margin (float): Margin value for margin loss.
    
    Returns:
        torch.Tensor: Combined loss value for the generator.
    """
    gen_adv_loss = adv_loss(discriminator(fake_data), real=True)
    real_embed = encoder(real_data)
    fake_embed = encoder(fake_data)
    gen_embed_loss = embed_loss(real_embed, real_embed, fake_embed, fake_embed)
    gen_margin_loss = margin_loss(real_embed, fake_embed, delta=margin)
    return gen_adv_loss + gen_embed_loss + gen_margin_loss
