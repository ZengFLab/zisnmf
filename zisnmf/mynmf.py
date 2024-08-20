import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], idx
    
    
class MyNMF(nn.Module):
    def __init__(self, n_components, device=None):
        super(MyNMF, self).__init__()
        self.n_components = n_components
        self.W = None
        self.H = None

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

    def fit_transform(self, X, num_epochs=200, learning_rate=0.01, alpha=0.2, batch_size=128, tol=1e-4):
        device = self.device
        X = torch.tensor(X, dtype=torch.float32).to(device)

        # Create DataLoader for batch processing
        dataset = CustomDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize W and H
        m, n = X.shape
        self.W = torch.rand((m, self.n_components), dtype=torch.float32, device=device, requires_grad=True)
        self.H = torch.rand((self.n_components, n), dtype=torch.float32, device=device, requires_grad=True)

        optimizer = optim.Adam([self.H], lr=learning_rate)

        # Training loop
        with tqdm(total=num_epochs, desc='Training', unit='epoch') as pbar:
            prev_loss = float('inf')
            for epoch in range(num_epochs):
                epoch_loss = 0.0

                for x_batch, batch_indices in dataloader:
                    optimizer.zero_grad()

                    # Calculate batch-specific WH
                    W_batch = self.W[batch_indices].clone().detach().requires_grad_(True)

                    WH = torch.mm(W_batch, self.H)
                    loss = torch.norm(x_batch - WH, p=2)**2  
                    
                    loss += alpha * torch.norm(self.H, p=1)
                    loss += alpha * torch.norm(W_batch, p=1)

                    tmp_HH = torch.mm(self.H, self.H.T)
                    loss += alpha * torch.norm(tmp_HH-torch.eye(tmp_HH.shape[0]).to(device), p=2)**2

                    loss.backward()

                    # Manually update W_batch
                    with torch.no_grad():
                        self.W[batch_indices] -= learning_rate * W_batch.grad

                    optimizer.step()

                    # Enforce non-negativity
                    self.W.data.clamp_(min=0)
                    self.H.data.clamp_(min=0)

                    epoch_loss += loss.item()

                epoch_loss /= len(dataloader)

                # Update progress bar
                pbar.set_postfix({'loss': epoch_loss})
                pbar.update(1)

                # Check for convergence
                if abs(prev_loss - epoch_loss) < tol:
                    break
                prev_loss = epoch_loss

        self.W = self.W.detach()
        self.H = self.H.detach()
        return self.W.cpu().numpy(), self.H.cpu().numpy()

    def transform_(self, X, num_epochs=100, learning_rate=0.01):
        X = X.to(self.device)
        H = self.H.to(self.device)
        
        # Initialize W for new data
        W_new = torch.rand(X.shape[0], self.n_components, device=self.device, requires_grad=True)
            
        # Optimize W while keeping H fixed
        optimizer = torch.optim.Adam([W_new], lr=learning_rate)
            
        if True:
            for _ in range(num_epochs):  # Adjust the number of iterations as needed
                # Compute the reconstructed matrix
                X_reconstructed = torch.mm(W_new, H)
                
                # Compute the loss
                loss = torch.norm(X-X_reconstructed, p=2) ** 2
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Project W_new to nonnegative space
                W_new.data.clamp_(0)

        return W_new
    
    def transform(self, X, num_epochs=100, learning_rate=0.01, batch_size=64):
        X = X.to(self.device)

        # Calculate the number of batches
        num_batches = (X.size(0) + batch_size - 1) // batch_size  # This ensures we cover all samples
        W_new = []
        with tqdm(total=num_batches, desc='Transforming', unit='batch') as pbar:
            for batch_idx in range(num_batches):  # Iterate over batches
                # Get the batch data
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, X.size(0))
                X_batch = X[start_idx:end_idx]  # Get the current batch

                W_batch = self.transform_(X_batch, num_epochs, learning_rate)

                W_new.append(W_batch)

                # Update progress bar
                pbar.update(1)
        
        W_new = torch.concat(W_new)
        W_new = W_new.detach().cpu().numpy()
        return W_new
    
    def inverse_transform(self, W):
        return torch.mm(torch.tensor(W, dtype=torch.float32), self.H.cpu()).numpy()

