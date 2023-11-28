import torch.nn as nn
import torch

# An adaptation of the original code by
# https://github.com/AlexanderMath/fasth/blob/master/fasthpp.py
class HouseHolder(nn.Module):
    """ Custom rotation layer using cayley transform to parametrize """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.weights = nn.Parameter(torch.eye(dim))
        self.pad = (1<<(dim-1).bit_length()) - dim
        self.p_dim = self.dim + self.pad
        self.log2dim = (1<<(dim-1).bit_length()).bit_length()-1
    
    def pad_X(self, X):
        return nn.functional.pad(X, (0, 0, 0, self.pad), "constant", 0)
    
    def unpad_X(self, X):
        if self.pad:
            return X[:-self.pad, :]
        else:
            return X
    
    def get_V(self):
        V = nn.functional.pad(
            nn.functional.normalize(self.weights.clone(), dim=0),
            (0, self.pad, 0, self.pad), "constant", 0
        )
        if self.pad:   
            V[-self.pad:, -self.pad:] += torch.eye(self.pad)
        return V
    
    def fasthpp(self, X):
        V = self.get_V()
        Y_ = V.clone().T
        W_ = -2*Y_.clone()

        k = 1
        for _ in range(self.log2dim):  
            k_2 = k
            k  *= 2

            W_view = W_.view(self.p_dim//k_2, k_2, self.p_dim).clone()
            m1_ = Y_.view(self.p_dim//k_2, k_2, self.p_dim)[0::2] @ torch.transpose(W_view[1::2], 1, 2)
            m2_ = torch.transpose(W_view[0::2], 1, 2) @ m1_

            W_ = W_.view(self.p_dim//k_2, k_2, self.p_dim)
            W_[1::2] += torch.transpose(m2_, 1, 2)
            W_ = W_.view(self.p_dim, self.p_dim)

        return X + self.unpad_X(W_.T @ (Y_ @ self.pad_X(X)))


    def forward(self, X):
        return self.fasthpp(X)

class Model(nn.Module):

    NAME = "FAST_HPP"

    def __init__(self, number_of_bits):
        super().__init__()
        self.number_of_bits = number_of_bits
        self.rot = HouseHolder(number_of_bits)

    def forward(self, X):
        return self.rot(X.T).T