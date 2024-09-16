from mytorch import Model, Parameter, Tensor
import numpy as np

class Embedding(Model):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(np.empty((num_embeddings, embedding_dim)))
        self.reset_parameters()
        
    def reset_parameters(self):
        self.weight.data = np.random.randn(self.num_embeddings, self.embedding_dim)
        
    def forward(self, x: Tensor):
        return self.weight[x]