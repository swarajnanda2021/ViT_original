
import torch
import torch.nn as nn

# We will now write a script for getting image patches

class PatchEmbed(nn.Module):

    '''
    Params:
    -------
    img_size (int)      : size of the image (assumed to be square) 
    patch_size (int)    : size of the patch (assumed to be square)
    in_chans (int)      : number of channels in the image (assumed to be RGB typically)
    embed_dim (int)     : embedding dimension (will be constant throughout the network)
    
    Attributes:
    -----------
    num_patches (int)   : number of patches in the image
    proj (nn.Conv2d)    : convolutional layer to get the patches, will have same stride as patch_size
    '''
    def __init__(self, img_size, patch_size, in_chans=3,embed_dim=256):
        super().__init__() # call the super class constructor which is used to inherit the properties of the parent class
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2 # assuming square image
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride = patch_size
        )
    
    def forward(self, x):
        ''' Parameters: 
        x (torch.Tensor): input image of shape (n_samples or batches, number of channels, height, width)
        Returns: 
        output = n_samplex X n_patches X embed_dim shape tensor
        '''
        x = self.proj(x) # n_samples X embed_dim X sqrt(n_patches) X sqrt(n_patches)
        x = x.flatten(2) # n_sample X embed_dim X n_patches
        x = x.transpose(1, 2) # n_samples X n_patches X embed_dim (dimensions are swapped)

        return x


# Let us now write the attention module
class Attention(nn.Module):
    ''' 
    Parameters
    ----------
    dim (int)           : embedding dimension, 
    n_heads (int)       : number of attention heads
    qkv_bias (bool)     : if True, we will include a bias in the query, key and value projections
    attn_d (float)      : Probability of dropout added to q, k and v during the training
    proj_d (float)      : Probability of dropout added to the projection layer
    
    Attributes
    __________
    scale (float)               : Used for norrmalizing the dot product
    qkv (nn.Linear)             : Linear projection, which are used for performing the attention
    proj (nn.Linear)            : Takes in the concatenated output of all attention heads and maps it further
    attn_d, proj_d (nn.Dropout) : Dropout layers

    '''
    def __init__(self,dim, n_heads=4, qkv_bias = False, attn_d = 0., proj_d = 0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5 # scaling added as per Vaswani paper for not feeding extremely large values to softmas
        self.qkv = nn.Linear(dim,dim * 3, bias = qkv_bias) # can be written separately too
        self.proj = nn.Linear(dim, dim)
        self.proj_d = nn.Dropout(proj_d)
        self.attn_d = nn.Dropout(attn_d)
    
    def forward(self,x):
        ''' 
        Parameters
        ----------
        x (torch.Tensor) : has shape (n_samples/batch, n_patches+1, dim)
        
        Returns
        -------
        torch.Tensor (n_samples, n_patches+1, dim)

        '''
        n_samples, n_tokens, dim = x.shape # extract shapes, tokens and dimensions from the output of the embeddings
        if dim != self.dim:
            raise ValueError # raise an error if dim isn't equal to the dimension set in the attention layer
        
        qkv = self.qkv(x) # Perform the query, key, value projections. (n_samples/batches, n_patches+1, 3*dim), the middle dimension is maintained

        # Let us now reshape the qkv tensor to separate the query, key and value
        qkv = qkv.reshape(n_samples,n_tokens,3,self.n_heads,self.head_dim) # (n_samples, n_patches+1, 3, n_heads, head_dim)
        qkv = qkv.permute(2,0,3,1,4) # (3, n_samples, n_heads, n_patches+1, head_dim)
        # Now extract the query, key and value
        q,k,v = qkv[0], qkv[1], qkv[2] # (n_samples, n_heads, n_patches+1, head_dim)
        # perform the dot product and scale the dot product
        dot_prod = (q @ k.transpose(-2,-1)) * self.scale # (n_samples, n_heads, n_patches+1, n_patches+1)
        # apply a softmax
        attention = dot_prod.softmax(dim = -1) # (n_samples, n_heads, n_patches+1, n_patches+1)
        attention = self.attn_d(attention) # apply dropout for regularization during training
        # weighted average
        wei = (attention @ v).transpose(1,2) # (n_samples, n_patches+1, n_heads, head_dim)
        # flatten
        wei = wei.flatten(2) # (n_samples, n_patches+1, dim) as dim = n_heads * head_dim
        # we now apply the projection
        x = self.proj(wei) # (n_samples, n_patches+1, dim)
        x = self.proj_d(x) # apply dropout for regularization during training
        return x

    # Let us now write the MLP module

class MLP(nn.Module):
    ''' 
    Parameters
    ----------
    in_features (int)           : embedding dimension, 
    hidden_features(int)        : dimension of the hidden layer
    out_features (int)          : dimension of the hidden layer
    dropout (float)     : probability of dropout
    
    Attributes
    __________
    fc1 (nn.Linear)     : Linear projection, which are used for performing the attention
    fc2 (nn.Linear)     : Takes in the concatenated output of all attention heads and maps it further
    dropout (nn.Dropout): Dropout layer

    '''
    def __init__(self, in_features,hidden_features,out_features, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features) # takes in the input and maps it to the hidden layer
        self.fc2 = nn.Linear(hidden_features, out_features) # takes in the hidden layer and maps it to the output
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU() # we will use the GELU activation function as per the paper
    
    def forward(self, x):
        x = self.fc1(x) # apply the first linear projection, (n_samples, n_patches+1, hidden_features)
        x = self.act(x) # apply the activation function (n_samples, n_patches+1, hidden_features)
        x = self.dropout(x) # apply dropout (n_samples, n_patches+1, hidden_features)
        x = self.fc2(x) # apply the second linear projection (n_samples, n_patches+1, out_features)
        x = self.dropout(x) # apply dropout (n_samples, n_patches+1, out_features)
        return x

# We have everything we need to write the ViT class

class Block(nn.Module):
    ''' Transformer with Vision Token
    Parameters
    ----------
    dim (int)           : embedding
    n_heads (int)       : number of attention heads
    mlp_ratio (float)   : ratio of mlp hidden dim to embedding dim, determines the hidden dimension size of the MLP module
    qkv_bias (bool)     : whether to add a bias to the qkv projection layer
    attn_d, proj_d,          : dropout probabilities

    Attributes
    ----------
    norm1, norm2        :  LayerNorm layers
    attn                : Attention layer
    mlp                 : MLP layer
    '''
    def __init__(self, dim, n_heads, mlp_ratio = 4.0, qkv_bias=True, attn_d=0., proj_d = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps = 1e-6) # division by zero is prevented and we match the props of the pretrained model
        self.attn = Attention(dim, n_heads, qkv_bias, attn_d, proj_d)
        self.norm2 = nn.LayerNorm(dim, eps = 1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features = dim, hidden_features = hidden_features, out_features = dim, dropout = proj_d)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x)) # add to the residual highway after performing Layernorm and attention
        x = x + self.mlp(self.norm2(x)) # add to the residual highway after performing Layernorm and MLP
        return x


# now we can write the Vision Transformer class
class ViT(nn.Module):
    ''' Vision Transformer
    Parameters
    ----------
    image_size (int)            : size of the input image
    patch_size (int)            : size of the patches to be extracted from the input image
    in_channels (int)           : number of input channels
    num_classes (int)           : number of classes
    embed_dim (int              : embedding dimension
    depth (int)                 : number of transformer blocks
    n_heads (int)               : number of attention heads per block
    mlp_ratio (float)           : ratio of mlp hidden dim to embedding dim, determines the hidden dimension size of the MLP module
    qkv_bias (bool)             : whether to add a bias to the qkv projection layer
    attn_d, proj_d,             : dropout probabilities

    Attributes
    ----------
    patch_embed (nn.Conv2d)     : Convolutional embedding layer
    pos_embed (nn.Parameter)    : learnable positional embedding
    cls_token (nn.Parameter)    : learnable class token
    blocks (nn.ModuleList)      : list of transformer blocks
    norm (nn.LayerNorm)         : final LayerNorm layer
    head (nn.Linear)            : final linear projection layer
    '''
    # initialize
    def __init__(self, 
                img_size = 384, 
                patch_size = 16, 
                in_chans=3, 
                n_classes = 1000, 
                embed_dim = 768, 
                depth = 12,
                n_heads = 12,
                mlp_ratio = 4.0,
                qkv_bias = True,
                attn_d = 0.,
                proj_d = 0.):
        super().__init__()
        # we will use the same image size as the pretrained model
        self.patch_embed = PatchEmbed(
                                        img_size = img_size,
                                        patch_size = patch_size,
                                        in_chans = in_chans,
                                        embed_dim = embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim)) # learnable class token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim)) # learnable positional embedding
        self.pos_d     = nn.Dropout(p = proj_d) # dropout layer
        self.blocks    = nn.ModuleList(
                            [
                                Block( 
                                    dim = embed_dim, 
                                    n_heads = n_heads, 
                                    mlp_ratio = mlp_ratio, 
                                    qkv_bias = qkv_bias, 
                                    attn_d = attn_d, 
                                    proj_d = proj_d) for _ in range(depth)] # iteratively create the transformer blocks with same parameters
                                    )
        self.norm       = nn.LayerNorm(embed_dim, eps = 1e-6) # final LayerNorm layer
        self.head       = nn.Linear(embed_dim, n_classes) # final linear projection layer    

    # forward pass
    def forward(self, x):
        ''' Forward pass
        Parameters
        ----------
        x (torch.Tensor)            : n_samples X in_chans X img_size X img_size
        Returns
        -------
        logits (torch.Tensor)       : n_samples X n_classes
        '''
        n_samples = x.shape[0]
        x = self.patch_embed(x) # extract patches from the input image and turn them into patch embeddings
        cls_tokens = self.cls_token.expand(n_samples, -1, -1) # expand the class token to match the batch size
        # pre-append the class token to the patch embeddings
        x = torch.cat((cls_tokens, x), dim = 1) # n_samples X (n_patches + 1) X embed_dim
        x = x + self.pos_embed # add the positional embedding to the patch embeddings
        x = self.pos_d(x) # apply dropout to the embeddings
        for block in self.blocks: # apply transformer blocks
            x = block(x)
        x = self.norm(x) # apply LayerNorm to the final output
        # the shape of x now is n_samples X (n_patches + 1) X embed_dim
        # extract the class token from the output
        cls_token = x[:, 0] # n_samples X embed_dim
        x = self.head(cls_token) # n_samples X n_classes
        return x
