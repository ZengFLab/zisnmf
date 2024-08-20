import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from .statsmodel import zig_nll
from .statsmodel import DropoutRate

import numpy as np
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, X, L):
        self.X = X
        self.L = L

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.L[idx], idx
    
# Convolutional Neural Network model
class MyCNNClassifier(nn.Module):
    def __init__(self, input_length, num_classes, input_channels):
        super(MyCNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1)  # Convolutional layer
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)  # Max pooling layer
        self.fc1_input_size = 64 * (input_length // 4)  # Adjusted input size for fc1
        self.fc1 = nn.Linear(self.fc1_input_size, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, num_classes)  # Output layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Dropout layer

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension (batch_size, channels, length)
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # Shape: (batch_size, 32, length/2)
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # Shape: (batch_size, 64, length/4)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))  # Fully connected layer
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Output layer
        return x
    
class MyMLPClassifier(nn.Module):
    def __init__(self, n_genes, n_classes, hidden_dims=[64]):
        super(MyMLPClassifier, self).__init__()
        
        # Define the layers with batch normalization and dropout
        self.layers = nn.ModuleList()
        input_dim = n_genes
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        mu = self.fc(x)
        return mu
    

class ZISNMF(nn.Module):
    def __init__(self, n_cells, n_features, n_states, n_classes, zero_inflated=True, 
                 hidden_dims=[64], init_type='random', random_seed=42, delta=0.0, 
                 classify_method = 'linear', input_channels = 1,
                 device=None):
        super(ZISNMF, self).__init__()
        self.n_cells = n_cells
        self.n_features = n_features
        self.n_states = n_states
        self.n_classes = n_classes
        self.init_type = init_type
        self.random_seed = random_seed
        self.delta = delta
        self.zero_inflated = zero_inflated
        self.classify_method = classify_method
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Set random seed for reproducibility
        torch.manual_seed(self.random_seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(self.random_seed)
        
        self.W = nn.Parameter(torch.empty(n_cells, n_states, device=self.device))
        self.H = nn.Parameter(torch.empty(n_states, n_features, device=self.device))

        # Define the layers for dropout rate
        self.dropout_rate = DropoutRate(n_features, hidden_dims=hidden_dims).to(self.device)
        self.sigma = nn.Parameter(torch.ones(1, 1, device=self.device))
        
        if self.classify_method == 'cnn':
            self.classifier = MyCNNClassifier(n_features, n_classes, input_channels).to(self.device)
        elif self.classify_method == 'linear':
            self.classifier = nn.Linear(n_features, n_classes).to(self.device)
        else:
            self.classifier = MyMLPClassifier(n_features, n_classes, hidden_dims).to(self.device)

    def _init_factors(self, X):
        if self.init_type == 'random':
            nn.init.xavier_uniform_(self.W)
            nn.init.xavier_uniform_(self.H)
        elif self.init_type == 'nndsvd':
            W, H = self._nndsvd_init(X)
            self.W.data.copy_(W)
            self.H.data.copy_(H)
        else:
            raise ValueError(f"Invalid initialization type: {self.init_type}")
        self.W.data.clamp_(0)
        self.H.data.clamp_(0)

    def _nndsvd_init(self, X):
        U, S, Vt = torch.linalg.svd(X)
        V = Vt.t()
        W = torch.zeros(self.n_cells, self.n_states, device=self.device)
        H = torch.zeros(self.n_states, self.n_features, device=self.device)

        W[:, 0] = torch.sqrt(S[0]) * torch.abs(U[:, 0])
        H[0, :] = torch.sqrt(S[0]) * torch.abs(V[:, 0])

        for j in range(1, self.n_states):
            x, y = U[:, j], V[:, j]
            x_p, y_p = torch.where(x > 0, x, torch.zeros_like(x)), torch.where(y > 0, y, torch.zeros_like(y))
            x_n, y_n = torch.abs(torch.where(x < 0, x, torch.zeros_like(x))), torch.abs(torch.where(y < 0, y, torch.zeros_like(y)))

            x_p_nrm, y_p_nrm = torch.norm(x_p), torch.norm(y_p)
            x_n_nrm, y_n_nrm = torch.norm(x_n), torch.norm(y_n)

            m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

            if m_p > m_n:
                u, v = x_p / x_p_nrm, y_p / y_p_nrm
                sigma = m_p
            else:
                u, v = x_n / x_n_nrm, y_n / y_n_nrm
                sigma = m_n

            W[:, j] = torch.sqrt(S[j] * sigma) * u
            H[j, :] = torch.sqrt(S[j] * sigma) * v

        return W, H

    def forward(self, X, L, W_batch):
        if L.shape[1] < self.n_states:
            W_masked = torch.zeros_like(W_batch)
            W_masked[:,:L.shape[1]] = W_batch[:,:L.shape[1]] * (L+self.delta)
            W_masked[:,L.shape[1]:] = W_batch[:,L.shape[1]:]
        else:
            W_masked = W_batch * (L+self.delta)
        X_reconstructed = torch.matmul(W_masked, self.H)
        return X_reconstructed
    
    def classify_loss(self, x_reconstructed, L):
        class_logits = self.classifier(x_reconstructed)
        class_loss = F.cross_entropy(class_logits, L)
        return class_loss

    def loss_function(self, X, X_reconstructed):
        if self.zero_inflated:
            pi = self.dropout_rate(X)
            self.sigma.data.clamp_(0.1)
            loss = zig_nll(X, pi, X_reconstructed, self.sigma)
        else:
            loss = torch.norm(X-X_reconstructed, p=2) ** 2

        return loss

    def fit(self, X, L, num_epochs=30, learning_rate=0.001, alpha=0.2, batch_size=1024, patience=10):
        X = X.to(self.device)
        L = L.to(self.device)
        
        dataset = CustomDataset(X, L)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self._init_factors(X)  # Initialize factors using the input matrix X
        
        # Exclude self.W from the optimizer
        optimizer = torch.optim.Adam([param for name, param in self.named_parameters() if name != 'W'], lr=learning_rate)

        best_loss = float('inf')
        epochs_without_improvement = 0

        with tqdm(total=num_epochs, desc='Training', unit='epoch') as pbar:
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                for batch_X, batch_L, batch_indices in dataloader:
                    batch_X = batch_X.to(self.device)
                    batch_L = batch_L.to(self.device)

                    lam = batch_X.shape[0] / X.shape[0]

                    # Slice and clone the corresponding rows of W for the current batch with requires_grad=True
                    W_batch = self.W[batch_indices].clone().detach().requires_grad_(True)

                    # Forward pass
                    X_reconstructed = self.forward(batch_X, batch_L, W_batch)

                    # Compute loss
                    reconstruct_loss = self.loss_function(batch_X, X_reconstructed)

                    W_loss = alpha * F.cross_entropy(W_batch[:,:batch_L.shape[1]], batch_L)

                    H_loss = 0
                    if batch_L.shape[1] < self.n_states:
                        tmp_H1H2 = lam * torch.matmul(self.H[:batch_L.shape[1],:], self.H[batch_L.shape[1]:,:].T)
                        H_loss = alpha * torch.norm(tmp_H1H2, p=2) ** 2
                        tmp_H2H2 = lam * torch.matmul(self.H[batch_L.shape[1]:,:], self.H[batch_L.shape[1]:,:].T)
                        H_loss += alpha * torch.norm(tmp_H2H2-torch.eye(tmp_H2H2.shape[0]).to(self.device), p=2) ** 2

                    if batch_L.shape[1] < self.n_states:
                        sparse_loss = alpha * torch.norm(W_batch[:,batch_L.shape[1]:], p=1)
                        sparse_loss += lam * alpha * torch.norm(self.H[batch_L.shape[1]:,:], p=1)
                    else:
                        sparse_loss = lam * alpha * torch.norm(self.H, p=1)
                    
                    X_reconstructed2 = torch.matmul(W_batch[:,:batch_L.shape[1]], self.H[:batch_L.shape[1],:])
                    class_loss = self.classify_loss(X_reconstructed2, batch_L)

                    L_predict = torch.matmul(batch_X, self.H[:batch_L.shape[1],:].T)
                    class_loss += F.cross_entropy(L_predict, batch_L)

                    total_loss = reconstruct_loss + class_loss + W_loss + H_loss + sparse_loss

                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()

                    # Manually update W_batch
                    with torch.no_grad():
                        self.W[batch_indices] -= learning_rate * W_batch.grad

                    # Step the optimizer to update other parameters
                    optimizer.step()

                    # Project W, H, and B to nonnegative space
                    self.W.data.clamp_(0)
                    self.H.data.clamp_(0)

                    epoch_loss += total_loss.item()

                # Calculate average loss for the epoch
                epoch_loss /= len(dataloader)

                # Early stopping check
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

                # Update progress bar
                pbar.set_postfix({'loss': epoch_loss})
                pbar.update(1)

    def fit_transform(self, X, L, num_epochs=30, learning_rate=0.001, alpha=0.2, batch_size=1024, patience=10):
        self.fit(X, L, num_epochs, learning_rate, alpha, batch_size, patience)
        return self.get_factors()
    
    def transform_(self, X, num_epochs=100, learning_rate=0.01, zero_inflated=True):
        X = X.to(self.device)
        
        # Initialize W for new data
        W_new = torch.rand(X.shape[0], self.n_states, device=self.device, requires_grad=True)
            
        # Optimize W while keeping H fixed
        optimizer = torch.optim.Adam([W_new], lr=learning_rate)
            
        if True:
            for _ in range(num_epochs):  # Adjust the number of iterations as needed
                # Compute the reconstructed matrix
                X_reconstructed = torch.mm(W_new, self.H)
                
                # Compute the loss
                if zero_inflated:
                    pi = self.dropout_rate(X)
                    loss = zig_nll(X, pi, X_reconstructed, self.sigma)
                else:
                    loss = torch.norm(X-X_reconstructed, p=2) ** 2
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Project W_new to nonnegative space
                W_new.data.clamp_(0)

        return W_new

    def transform(self, X, num_epochs=100, learning_rate=0.01, batch_size=64, zero_inflated=True):
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

                W_batch = self.transform_(X_batch, num_epochs, learning_rate, zero_inflated)

                W_new.append(W_batch)

                # Update progress bar
                pbar.update(1)
        
        W_new = torch.concat(W_new)
        W_new = W_new.detach().cpu().numpy()
        return W_new
    
    def inverse_transform(self, W):
        return torch.mm(torch.tensor(W, dtype=torch.float32), self.H[:W.shape[1],:].cpu()).numpy()
    
    def predict_proba(self, X, num_epochs=100, learning_rate=0.01, V_only=True, batch_size=64, zero_inflated=True):
        y_scores = []

        # Calculate the number of batches
        num_batches = (X.size(0) + batch_size - 1) // batch_size  # This ensures we cover all samples

        with tqdm(total=num_batches, desc='Predicting', unit='epoch') as pbar:
            for batch_idx in range(num_batches):  # Iterate over batches
                # Get the batch data
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, X.size(0))
                X_batch = X[start_idx:end_idx]  # Get the current batch

                W = self.transform_(X_batch, num_epochs=num_epochs, learning_rate=learning_rate, zero_inflated=zero_inflated)
                if V_only:
                    X_new = torch.mm(W[:,:self.n_classes], self.H[:self.n_classes,:])
                else:
                    X_new = torch.mm(W, self.H)

                y_scores.append(self.classifier(X_new))

                # Update progress bar
                pbar.update(1)

        y_scores = torch.concat(y_scores)
        return y_scores.detach().cpu().numpy()
    
    def predict(self, X, num_epochs=100, learning_rate=0.01, V_only=True, batch_size=64, zero_inflated=True):
        y_scores = self.predict_proba(X, num_epochs, learning_rate, V_only, batch_size, zero_inflated)
        y = np.argmax(y_scores, axis=1)
        return y

    def get_factors(self):
        return self.W.detach().cpu().numpy(), self.H.detach().cpu().numpy()
    
