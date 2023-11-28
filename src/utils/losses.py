import torch
import numpy as np


# DSH loss
# Paper: Deep Supervised Hashing for Fast Image Retrieval
# Authors: Haomiao Liu, Ruiping Wang, Shiguang Shan, Xilin Chen.
# https://ieeexplore.ieee.org/document/7780596/
def DSH(separation=24, quantization_penalty=0.01):
    ReLU = torch.nn.ReLU()
    
    def loss(Z, label):
        sm = (label @ label.T > 0).long()
        dists = torch.sum((Z[:,None,:]-Z[None,:,:])**2, dim = -1)
        similarity_term = torch.mean(.5*sm*dists + .5*(1-sm)*ReLU(separation - dists))
        quantization_term = quantization_penalty*torch.mean(torch.sum(1 - torch.abs(Z), dim = -1))

        return similarity_term + quantization_term

    return loss


# DPSH loss
# Paper: Feature Learning based Deep Supervised Hashing with Pairwise Labels
# Authors: Wu-Jun Li, Sheng Wang, Wang-Cheng Kang
# https://arxiv.org/abs/1511.03855
def DPSH(quantization_penalty=0.01):
    LogSigmoid = torch.nn.LogSigmoid()

    def loss(Z, label):
        sm = (label @ label.T > 0).long()
        sims = Z @ Z.t()
        similarity_term = -torch.mean( sm*sims + LogSigmoid(-sims) )
        quantization_term = quantization_penalty*torch.mean(torch.sum( (Z - torch.sign(Z))**2, dim = -1))

        return similarity_term + quantization_term

    return loss


# DHN loss
# Paper: Deep hashing network for efficient similarity retrieval.
# Authors: H. Zhu, M. Long, J. Wang, Y. Cao
# https://ojs.aaai.org/index.php/AAAI/article/view/10235
def DHN(quantization_penalty=0.01):
    LogSigmoid = torch.nn.LogSigmoid()

    def loss(Z, label):
        sm = (label @ label.T > 0).long()
        sims = Z @ Z.t()
        similarity_term = -torch.mean( sm*sims + LogSigmoid(-sims) )
        quantization_term = quantization_penalty*torch.mean(torch.sum( torch.log(torch.cosh(torch.abs(Z))) - 1 , dim = -1))

        return similarity_term + quantization_term

    return loss


# DCH loss
# Paper: Deep cauchy hashing for hamming space retrieval.
# Authors: Y. Cao, M. Long, B. Liu, J. Wang. 
# https://openaccess.thecvf.com/content_cvpr_2018/html/Cao_Deep_Cauchy_Hashing_CVPR_2018_paper.html
def cosdist(x,y):
    cossim = torch.nn.CosineSimilarity()
    eps = 1e-5
    return eps + (1 - cossim( x, y ))/2*(1 - 2*eps)

def DCH(gamma=10, quantization_penalty=0.01, p=.5):
    def loss(Z, label):
        c = Z.shape[-1]
        sm = (label @ label.T > 0).float()
        w = sm/p + (1-sm)/(1-p)
        d = c*cosdist( Z[:, :, None], Z.t()[None, :, :] )
        similarity_term = torch.mean( w*( sm*torch.log(d/gamma) + torch.log(1+gamma/d) ) )
        quantization_term = quantization_penalty*torch.mean(torch.log( 1 + (c*cosdist(torch.abs(Z), Z**0))/gamma ))

        return similarity_term + quantization_term

    return loss


# WGLHH loss
# Paper: Weighted gaussian loss based hamming hashing. 
# Authors:  R.-C. Tu, X.-L. Mao, C. Kong, Z. Shao, Z.-L. Li, W. Wei, and H. Huang. W
# https://dl.acm.org/doi/10.1145/3474085.3475498
def WGLHH(alpha=0.1, quantization_penalty=0.01, p=.5):
    cossim = torch.nn.CosineSimilarity()
    LogSigmoid = torch.nn.LogSigmoid()

    def loss(Z, label):
        sm = (label @ label.T > 0).float()
        w = sm/p + (1-sm)/(1-p)
        cs = cossim( Z[:, :, None], Z.t()[None, :, :] )
        d = Z.shape[-1]*(1 - cs)/2
        a = torch.exp(torch.abs(.5*(cs+1) - sm))
        x = alpha*d**2
        ex = torch.exp(-x)

        similarity_term = torch.mean(w*a*((1-sm)*ex + sm*((1+ex)*LogSigmoid(x) - x*ex)))
        quantization_term = quantization_penalty*torch.mean(torch.sum( (Z - torch.sign(Z))**2, dim = -1))

        return similarity_term + quantization_term

    return loss


# HashNet loss
# Paper: HashNet: Deep Learning to Hash by Continuation
# Authors: Cao, Zhangjie and Long, Mingsheng and Wang, Jianmin and Yu, Philip S
# https://arxiv.org/abs/1702.00758
def HashNet(p=.5):
    LogSigmoid = torch.nn.LogSigmoid()

    def loss(Z, label):
        sm = (label @ label.T > 0).float()
        sims = Z @ Z.t()
        w = sm/p + (1-sm)/(1-p)
        alpha = 10/Z.shape[-1]
        similarity_term = -torch.mean(w*( alpha*sm*sims + LogSigmoid(-alpha*sims) ))
        
        return similarity_term

    return loss


# HyP² loss
# Paper: HyP² Loss: Beyond Hypersphere Metric Space for Multi-label Image Retrieval
# Authors: C Xu, Z Chai, Z Xu, C Yuan, Y Fan, J Wang
# https://dl.acm.org/doi/pdf/10.1145/3503161.3548032
def HyP2_pair(separation = 0.1):
    cossim = torch.nn.CosineSimilarity()
    ReLU = torch.nn.ReLU()

    def loss(Z, label):
        cos = cossim( Z[:, :, None], Z.t()[None, :, :] )
        smc = (label @ label.T <= 0).float()
        cos_minus = ReLU(cos - separation)
        similarity_term = torch.mean(smc*cos_minus)/torch.mean(smc)

        return similarity_term

    return loss

def HyP2_proxy(separation = 0.1):
    cossim = torch.nn.CosineSimilarity()
    ReLU = torch.nn.ReLU()

    def loss(Z, label, P):
        cos = cossim( Z[:, :, None], P.t()[None, :, :] )
        cos_minus = ReLU(cos - separation)
        cos_plus = -cos
        similarity_term = torch.mean(label*cos_plus)/torch.mean(label)
        dissimilarity_term = torch.mean((1-label)*cos_minus)/torch.mean(1-label)
        
        return similarity_term + dissimilarity_term

    return loss


# Cosine Embedding Loss
# As in https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html
def CEL(separation = 0.1):
    cossim = torch.nn.CosineSimilarity()
    ReLU = torch.nn.ReLU()

    def loss(Z, label):
        sm = (label @ label.T > 0).float()
        cos = cossim( Z[:, :, None], Z.t()[None, :, :] )
        cos_minus = ReLU(cos - separation)
        similarity_term = torch.mean((1-cos)*sm + (1-sm)*cos_minus)

        return similarity_term

    return loss


# HSWD loss
# adapted from https://github.com/khoadoan106/single_loss_quantization/blob/main/python/losses/distributional_quantization_losses.py
def wasserstein1d(x, y, aggregate=True):
    """Compute wasserstein loss in 1D"""
    x1, _ = torch.sort(x, dim=0)
    y1, _ = torch.sort(y, dim=0)
    n = x.size(0)
    if aggregate:
        z = (x1-y1).view(-1)
        return torch.dot(z, z)/n
    else:
        return (x1-y1).square().sum(0)/n
        
def HSWD(b, device='cuda', aggregate=True):
    real_b = torch.randn(b.shape, device=device).sign()
    bsize, dim = b.size()

    if aggregate:
        gloss = wasserstein1d(real_b, b) / dim
    else:
        gloss = wasserstein1d(real_b, b, aggregate=False)

    return gloss


# Bit var loss for rotation
def bit_var_loss(dist="logistic"):
    if dist=="logistic":
        def F(x):
            return 1/(1+torch.exp(-x))

    def loss(Z,label):
        return torch.mean(F(Z)*(1-F(Z)))
    return loss

# Closes sphere bit loss for rotation
def CSB():
    def loss(Z, label):
        k = Z.shape[-1]
        sphere_bits = torch.linalg.norm(Z,axis=1)/np.sqrt(k)
        return torch.sum((torch.abs(Z)-sphere_bits.repeat([k,1]).t())**2)
    return loss

# Mean abs loss for rotation
def MA(p=2):
    def loss(Z, label, p = p):
        k = Z.shape[-1]
        center = torch.mean(torch.abs(Z),axis=-1)
        return torch.sum(torch.abs(Z-center.repeat([k,1]).t())**p)
        
    return loss

# L2 loss for rotation
def L2():
    def loss(Z, label):
        return torch.mean((Z-torch.sgn(Z))**2)
    return loss

# L1 loss for rotation
def L1():
    def loss(Z, label):
        return torch.mean(torch.abs(Z-torch.sgn(Z)))
    return loss

# cosine similarity between Z and sgn(Z) loss for rotation
def cos_sim():
    def loss(Z, label):
        return -torch.mean(torch.abs(Z))
    return loss

# cosine similarity between Z and sgn(Z) loss for rotation
def min_entry():
    def loss(Z, label):
        return torch.mean(torch.logsumexp(-Z**2, dim=-1))
    return loss

# Pertubed collision loss
def perturbed_collision_probability(noise_distribution='Normal', non_matches_noise_std = 0.4, matches_noise_std = 0.4, agg_type="joint", matches_repulsion=-1, default_balance = 0.5):

    if noise_distribution=='Normal':
        F = torch.distributions.normal.Normal(0,1).cdf
    if noise_distribution=='Uniform':
        F = torch.distributions.uniform.Uniform(-1.73,1.73).cdf
    if noise_distribution=='Cauchy':
        F = torch.distributions.cauchy.Cauchy(0,1).cdf
    if noise_distribution=='Logistic':
        F = torch.nn.Sigmoid()

    def ploss(Z, label, balance=default_balance):
        sm = (label @ label.T > 0).float()
        p = torch.mean(sm)
        w = balance*sm/p + (1-balance)*(1-sm)/(1-p)

        pos_m = F(1/matches_noise_std*Z)
        neg_m = F(-1/matches_noise_std*Z)
        pos_nm = F(1/non_matches_noise_std*Z)
        neg_nm = F(-1/non_matches_noise_std*Z)

        matches_loss = sm * torch.prod( pos_m[:, None, :] * pos_m[None, :, :] + neg_m[:, None, :] * neg_m[None, :, :], dim=-1)
        non_matches_loss = (1-sm) * torch.prod( pos_nm[:, None, :] * pos_nm[None, :, :] + neg_nm[:, None, :] * neg_nm[None, :, :], dim=-1)

        loss =  torch.mean(w*(non_matches_loss + matches_repulsion * matches_loss))
        return loss
    
    if agg_type=="joint":
        return ploss
    

def EH(r=2, p=1, std=.4):

    F = torch.distributions.normal.Normal(0,1).cdf
    ReLU = torch.nn.ReLU()

    def loss(Z, label):
        sm = (label @ label.T > 0).float()
        p = torch.mean(sm)
        k = Z.shape[-1]
        w = sm/p + (1-sm)/(1-p)

        pos_m = F(1/std*Z)
        neg_m = F(-1/std*Z)
        pos_nm = F(1/std*Z)
        neg_nm = F(-1/std*Z)

        matches_loss = sm * ReLU(k - r - torch.sum( pos_m[:, None, :] * pos_m[None, :, :] + neg_m[:, None, :] * neg_m[None, :, :], dim=-1))
        non_matches_loss = (1-sm) * torch.sum( pos_nm[:, None, :] * pos_nm[None, :, :] + neg_nm[:, None, :] * neg_nm[None, :, :], dim=-1)

        # breakpoint()

        loss =  torch.mean( w*(non_matches_loss + matches_loss) )
        return loss
    return loss






