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
        x = x.view(x.shape[0], -1)  # Flatten the tensor
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
        x = self.fc(x)
        return x
    

class ZISNMF(nn.Module):
    def __init__(self, n_cells, n_features, n_classes, n_extra_states, zero_inflated=True, 
                 hidden_dims=[64], random_seed=42, delta=0.0, 
                 classify_method = 'linear', input_channels = 1,
                 device=None):
        super(ZISNMF, self).__init__()
        self.n_cells = n_cells
        self.n_features = n_features
        self.n_extra_states = n_extra_states
        self.n_classes = n_classes
        self.n_states = n_classes + n_extra_states
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
        
        self.W = nn.Parameter(torch.empty(n_cells, self.n_states, device=self.device))
        self.H = nn.Parameter(torch.empty(self.n_states, n_features, device=self.device))

        # Define the layers for dropout rate
        self.dropout_rate = DropoutRate(n_features, hidden_dims=hidden_dims).to(self.device)
        self.sigma = nn.Parameter(torch.ones(1, 1, device=self.device))
        
        if self.classify_method == 'cnn':
            self.classifier = MyCNNClassifier(n_features, n_classes, input_channels).to(self.device)
        elif self.classify_method == 'linear':
            self.classifier = nn.Linear(n_features, n_classes).to(self.device)
        else:
            self.classifier = MyMLPClassifier(n_features, n_classes, hidden_dims).to(self.device)

    def _init_factors(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.H)
        self.W.data.clamp_(0)
        self.H.data.clamp_(0)

    def forward(self, L, W_batch):
        if self.n_extra_states>0:
            W_masked = torch.zeros_like(W_batch)
            W_masked[:,:self.n_classes] = W_batch[:,:self.n_classes] * (L+self.delta)
            W_masked[:,self.n_classes:] = W_batch[:,self.n_classes:]
        else:
            W_masked = W_batch * (L+self.delta)
        X_reconstructed = torch.matmul(W_masked, self.H)
        return X_reconstructed
    
    def classify_loss(self, x_reconstructed, L):
        class_logits = self.classifier(x_reconstructed)
        #class_loss = F.cross_entropy(class_logits, L)
        class_loss = nn.BCELoss()(nn.Sigmoid()(class_logits.reshape(-1)), L.reshape(-1))
        return class_loss

    def loss_function(self, X, X_reconstructed):
        if self.zero_inflated:
            pi = self.dropout_rate(X)
            self.sigma.data.clamp_(0.01)
            loss = zig_nll(X, pi, X_reconstructed, self.sigma)
        else:
            loss = torch.norm(X-X_reconstructed, p=2) ** 2

        return loss

    def fit(self, X, L, num_epochs=30, learning_rate=0.001, alpha=0.2, batch_size=1024, patience=10, algo='adam'):    
        dataset = CustomDataset(X, L)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self._init_factors()  # Initialize factors using the input matrix X
        
        # Exclude self.W from the optimizer
        if algo.lower() == 'adam':
            optimizer = torch.optim.Adam([param for name, param in self.named_parameters() if (name != 'W')], lr=learning_rate)
        elif algo.lower() == 'adamw':
            optimizer = torch.optim.AdamW([param for name, param in self.named_parameters() if (name != 'W')], lr=learning_rate)
        elif algo.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop([param for name, param in self.named_parameters() if (name != 'W')], lr=learning_rate)
        elif algo.lower() == 'sgd':
            optimizer = torch.optim.SGD([param for name, param in self.named_parameters() if (name != 'W')], lr=learning_rate)
        elif algo.lower() == 'asgd':
            optimizer = torch.optim.ASGD([param for name, param in self.named_parameters() if (name != 'W')], lr=learning_rate)
        else:
            raise ValueError("An optimization algorithm must be specified.")

        best_loss = float('inf')
        epochs_without_improvement = 0

        with tqdm(total=num_epochs, desc='Training', unit='epoch') as pbar:
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                for batch_X, batch_L, batch_indices in dataloader:
                    batch_X = move_to_device(batch_X,self.device)
                    batch_L = move_to_device(batch_L,self.device)

                    lam = batch_X.shape[0] / X.shape[0]

                    # Slice and clone the corresponding rows of W for the current batch with requires_grad=True
                    W_batch = self.W[batch_indices].clone().detach().requires_grad_(True)

                    # Forward pass
                    X_reconstructed = self.forward(batch_L, W_batch)

                    # Compute loss
                    reconstruct_loss = self.loss_function(batch_X, X_reconstructed)

                    M_loss = 0
                    M_loss += alpha * F.cross_entropy(W_batch[:,:self.n_classes], batch_L)

                    H_loss = 0
                    if self.n_extra_states>0:
                        tmp_H1H2 = lam * torch.matmul(self.H[:self.n_classes,:], self.H[self.n_classes:,:].T)
                        H_loss = alpha * torch.norm(tmp_H1H2, p=2) ** 2
                        tmp_H2H2 = lam * torch.matmul(self.H[self.n_classes:,:], self.H[self.n_classes:,:].T)
                        H_loss += alpha * torch.norm(tmp_H2H2-torch.eye(tmp_H2H2.shape[0]).to(self.device), p=2) ** 2

                    sparse_loss = 0
                    if self.n_extra_states>0:
                        sparse_loss = alpha * torch.norm(W_batch[:,self.n_classes:], p=1)
                        sparse_loss += lam * alpha * torch.norm(self.H[self.n_classes:,:], p=1)
                    
                    X_reconstructed2 = torch.matmul(W_batch[:,:self.n_classes], self.H[:self.n_classes,:])
                    class_loss = self.classify_loss(X_reconstructed2, batch_L)

                    # pattern match
                    L_predict = torch.matmul(batch_X, self.H[:self.n_classes,:].T)
                    class_loss += F.cross_entropy(L_predict, batch_L)

                    total_loss = reconstruct_loss + class_loss + M_loss + H_loss + sparse_loss

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

    def fit_transform(self, X, L, num_epochs=30, learning_rate=0.001, alpha=0.2, batch_size=1024, patience=10, algo='adam'):
        self.fit(X, L, num_epochs, learning_rate, alpha, batch_size, patience, algo=algo)
        return self.get_factors()
    
    def _transform(self, X, num_epochs=100, learning_rate=0.01, zero_inflated=True, algo='adam'):
        X = move_to_device(X,self.device)
        
        # Initialize W for new data
        W_new = torch.rand(X.shape[0], self.n_states, device=self.device, requires_grad=True)
            
        # Optimize W while keeping H fixed
        if algo.lower() == 'adam':
            optimizer = torch.optim.Adam([W_new], lr=learning_rate)
        elif algo.lower() == 'adamw':
            optimizer = torch.optim.AdamW([W_new], lr=learning_rate)
        elif algo.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop([W_new], lr=learning_rate)
        elif algo.lower() == 'sgd':
            optimizer = torch.optim.SGD([W_new], lr=learning_rate)
        elif algo.lower() == 'asgd':
            optimizer = torch.optim.ASGD([W_new], lr=learning_rate)
        else:
            raise ValueError("An optimization algorithm must be specified.")
            
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

        M_new = W_new[:,:self.n_classes]
        W_new = W_new[:,self.n_classes:]
        return M_new,W_new

    def transform(self, X, num_epochs=100, learning_rate=0.01, batch_size=64, zero_inflated=True):
        dataset = CustomDataset(X, X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        M_new = []
        W_new = []
        with tqdm(total=len(dataloader), desc='Transforming', unit='batch') as pbar:
            for X_batch, _, _ in dataloader:  # Iterate over batches
                M_batch,W_batch = self._transform(X_batch, num_epochs, learning_rate, zero_inflated)
                
                M_batch = tensor_to_numpy(M_batch)
                W_batch = tensor_to_numpy(W_batch)
                M_new.append(M_batch)
                W_new.append(W_batch)

                # Update progress bar
                pbar.update(1)
        
        M_new = np.concatenate(M_new)
        W_new = np.concatenate(W_new)
        return M_new,W_new
    
    def predict_proba(self, X, num_epochs=100, learning_rate=0.01, batch_size=64, zero_inflated=True):
        y_scores = []

        # Calculate the number of batches
        dataset = CustomDataset(X, X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with tqdm(total=len(dataloader), desc='Predicting', unit='epoch') as pbar:
            for X_batch, _, _ in dataloader:  # Iterate over batches
                M,_ = self._transform(X_batch, num_epochs=num_epochs, learning_rate=learning_rate, zero_inflated=zero_inflated)
                X_new = torch.mm(M, self.H[:self.n_classes,:])

                y_batch = self.classifier(X_new)
                y_batch = tensor_to_numpy(y_batch)
                y_scores.append(y_batch)

                # Update progress bar
                pbar.update(1)

        y_scores = np.concatenate(y_scores)
        return y_scores
    
    def predict(self, X, num_epochs=100, learning_rate=0.01, V_only=True, batch_size=64, zero_inflated=True):
        y_scores = self.predict_proba(X, num_epochs, learning_rate, V_only, batch_size, zero_inflated)
        y = np.argmax(y_scores, axis=1)
        return y

    def get_factors(self):
        M = tensor_to_numpy(self.W[:,:self.n_classes])
        V = tensor_to_numpy(self.H[:self.n_classes,:])

        if self.n_extra_states>0:
            W = tensor_to_numpy(self.W[:,self.n_classes:])
            H = tensor_to_numpy(self.H[self.n_classes:,:])
        else:
            W,H = None,None
        return M,V,W,H
    


def tensor_to_numpy(tensor):
    """
    Check if the tensor is on a CUDA device. If yes, detach it, move it to CPU,
    and convert to a NumPy array. If not, just detach and convert to NumPy.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        np.ndarray: The resulting NumPy array.
    """
    # Check if the input is a tensor
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a torch Tensor.")

    # Detach the tensor from the computation graph
    tensor = tensor.detach()
    
    # Check if the tensor is on CUDA
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Convert to NumPy
    numpy_array = tensor.numpy()
    return numpy_array

def move_to_device(data, device):
    """
    Checks if the input data is a tensor. If not, converts it to a tensor,
    checks if the tensor is on the specified device, and moves it if necessary.

    Args:
        data (any): The input data to check (can be a tensor, list, NumPy array, etc.).
        device (str or torch.device): The device to check against (e.g., 'cpu', 'cuda', 'cuda:0').

    Returns:
        torch.Tensor: The tensor on the specified device.
    """
    # Convert input data to tensor if it's not already a tensor
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)

    # Check if the device is a string, and convert it to torch.device if necessary
    device = torch.device(device) if isinstance(device, str) else device

    # Move the tensor to the specified device if necessary
    if data.device != device:
        data = data.to(device)
    
    return data