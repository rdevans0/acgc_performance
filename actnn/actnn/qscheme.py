import torch
import numpy as np

import actnn
from actnn.conf import config
import actnn.cpp_extension.minimax as ext_minimax
import actnn.cpp_extension.calc_precision as ext_calc_precision

from actnn.utils import to_cupy
import cupy

import pdb
try: 
    profile
except NameError:
    profile = lambda f:f

class QScheme(object):
    num_samples = 1
    num_layers = 0
    batch = None
    update_scale = True
    layers = []
    prev_layer = None
    ty = 'actnn'

    def __init__(self, layer, group=0, num_locations=1, depthwise_groups=1):
        self.initial_bits = config.initial_bits
        self.bits = config.activation_compression_bits[group]
        if config.use_gradient:
            assert QScheme.num_samples > 1
            self.scales = torch.zeros(QScheme.num_samples)
        else:
            self.scales = torch.tensor([0.0])
        QScheme.layers.append(self)
        self.C = None
        self.dim = None
        self.num_locations = num_locations      # Kernel size
        self.depthwise_groups = depthwise_groups    # Depthwise separable conv
        self.layer = layer
        self.layer_type = layer.__class__.__name__
        self.group = group

        # debug
        self.name = 'layer_{}'.format(QScheme.num_layers)
        QScheme.num_layers += 1
        
        print(f'{self.name} {QScheme.num_layers} {layer}')

    def get_scale(self):
        if config.use_gradient:
            assert QScheme.batch is not None
            scale = self.scales[QScheme.batch].clone()
            avg_scale = scale.mean()
            scale[scale == 0] = avg_scale + 1e-9
            return scale
        else:
            return self.scales

    def set_scale(self, grad):
        if QScheme.update_scale:
            if config.use_gradient:
                assert QScheme.batch is not None
                scale = grad.view(grad.shape[0], -1).float().norm(dim=1).square().cpu()
                self.scales[QScheme.batch] = self.scales[QScheme.batch] * 0.5 + scale * 0.5
            else:
                scale = grad.view(grad.shape[0], -1).float().norm(dim=1).square()
                self.scales = scale.mean()

    def compute_quantization_bits(self, input):
        QScheme.prev_layer = self
        N = input.shape[0]
        D = input.shape[1]
        input_flatten = input.view(N, -1)
        num_features = input_flatten.shape[1]
        num_pixels = num_features // D

        # Compute min, max by groups
        if num_features % config.group_size != 0:
            # Padding
            new_num_features = (num_features // config.group_size + 1) * config.group_size
            delta = new_num_features - num_features
            input_flatten = torch.cat([input_flatten,
                                       torch.zeros([N, delta], dtype=input.dtype, device=input.device)], 1)

        input_groups = input_flatten.view(-1, config.group_size)    # [-1, group_size]
        mn, mx = ext_minimax.minimax(input_groups)
        if not config.pergroup:    # No per group quantization
            mn = torch.ones_like(mn) * mn.min()
            mx = torch.ones_like(mx) * mx.max()

        # Average range over pixels     G * ||R_n||^2 / I
        Range_sqr = torch.norm((mx - mn).view(N, -1), dim=1).float().square() * (config.group_size / num_pixels)

        # greedy
        grad_sum = self.get_scale().cuda()
        C = (self.num_locations / 4 / self.depthwise_groups * Range_sqr * grad_sum)\
            .to(torch.float32).cpu()
        b = torch.ones(N, dtype=torch.int32) * self.initial_bits
        w = torch.ones(N, dtype=torch.int32)
        b = ext_calc_precision.calc_precision(b, C, w, int(self.bits * N))         # N

        self.C = C
        self.dim = input.numel() // N
        self.b = b

        return input_groups.view(N, -1, config.group_size), b.cuda(), mn.view(N, -1, 1), mx.view(N, -1, 1)
    
    @staticmethod
    def allocate_perlayer():
        num_groups = len(config.activation_compression_bits)
        for g in range(num_groups):
            layers = [layer for layer in QScheme.layers if layer.group == g]
            L = len(layers)

            if config.activation_compression_bits[g] == config.initial_bits:
                C = torch.tensor([layer.C.sum() for layer in layers])
                w = torch.tensor([layer.dim for layer in layers], dtype=torch.int)
                total_bits = w.sum() * config.activation_compression_bits[g]
                b = torch.ones(L, dtype=torch.int32) * 8
                b = ext_calc_precision.calc_precision(b, C, w, total_bits)

                for i in range(L):
                    layers[i].bits = layers[i].initial_bits = b[i]
            else:
                Cs = [layer.C for layer in layers]
                C = torch.cat(Cs, 0)

                N = Cs[0].shape[0]

                # TODO ???
                Ws = [torch.ones(N, dtype=torch.int32) * layer.dim for layer in layers]
                # Ws = [torch.ones(N, dtype=torch.int32) for layer in layers]
                w = torch.cat(Ws, 0)

                total_bits = w.sum() * config.activation_compression_bits[g]
                b = torch.ones(N * L, dtype=torch.int32) * config.initial_bits
                b = ext_calc_precision.calc_precision(b, C, w, total_bits)
                for i in range(L):
                    bs = b[i*N : (i+1)*N]
                    layers[i].bits = bs.float().mean()

    def if_allocate_perlayer(self):
        if not config.perlayer:
            return
        for layer in QScheme.layers:
            if layer.C is None:
                return

        first_layer = None
        for layer in QScheme.layers:
            if layer.layer.weight.requires_grad:
                first_layer = layer
                break

        # If myself is the last layer, then reallocate bits per layer
        if config.compress_activation and config.training:
            if self == first_layer:
                QScheme.allocate_perlayer()
                actnn.QBNScheme.allocate_perlayer()


class ACGC_FixedQScheme(object):
    auto = False
    ty = 'acgc'
    def __init__(self, layer, depthwise_groups=1, **kw):
        self.eps = 1
        
        QScheme.layers.append(self)
        self.layer = layer
        self.layer_type = layer.__class__.__name__
        
        # ACGC-Specific
        self.iteration = 0
        self.fs = []
        self.signed = None

        # debug
        self.name = 'layer_{}'.format(QScheme.num_layers)
        QScheme.num_layers += 1
        
        print(f'{self.name} {QScheme.num_layers} {layer}')


    def compute_quantization_bits(self, x):
        recalc_keep = ACGC_QScheme.recalc_keep
        
        if self.signed is None:
            self.signed = bool(x.min() < 0)
            
        if self.iteration % config.recalc_interval == 0:
            axis = [0,2,3] if x.ndim == 4 else None
            exp = (None, Ellipsis, None, None) if x.ndim == 4 else (None,)*x.ndim
            x_max = abs(x).max(axis=axis)
            f = (1.0 / (x_max + 1e-9) ) [exp]
            
            self.fs += [f]
            self.fs = self.fs[-recalc_keep:]
            self.cur_f = sum(self.fs) / len(self.fs)  # Summary by average
        
        else:
            self.saved_x_max = None
        
        self.iteration += 1
        per_channel = 'batchnorm' in self.layer_type.lower()
        bits = config.activation_compression_bits[0]
        return bits, self.cur_f, self.signed, per_channel
    
    def update_grads(*a, **kw):
        pass
    
    def if_allocate_perlayer(self):
        pass

class ACGC_QScheme(object):
    num_samples = 1
    num_layers = 0
    batch = None
    layers = []
    prev_layer = None
    recalc_keep = 10
    ty = 'acgc'
    auto = True

    def __init__(self, layer, depthwise_groups=1, **kw):
        self.eps = 1
        
        QScheme.layers.append(self)
        self.input_shape = None
        self.kernel_shape = None
        self.depthwise_groups = depthwise_groups    # Depthwise separable 
        assert depthwise_groups == 1, 'Unknown how to deal with depthwise'
        self.layer = layer
        self.layer_type = layer.__class__.__name__
        
        # ACGC-Specific
        self.iteration = 0
        self.bits = []
        self.fs = []
        self.error_bounds = []
        self.cur_error_bound = 2**-4 # Start at 4 bits for first iteration
        self.signed = None
        self.saved_x_max = None

        # debug
        self.name = 'layer_{}'.format(QScheme.num_layers)
        QScheme.num_layers += 1
        
        # print(f'{self.name} {QScheme.num_layers} {layer}')

    @profile
    def compute_quantization_bits(self, x):
        QScheme.prev_layer = self
        recalc_keep = ACGC_QScheme.recalc_keep
        self.shape = x.shape
        if self.signed is None:
            self.signed = bool(x.min() < 0)
        
        if self.iteration % config.recalc_interval == 0:
            axis = [0,2,3] if x.ndim == 4 else None
            exp = (None, Ellipsis, None, None) if x.ndim == 4 else (None,)*x.ndim
            x_max = abs(x).max(axis=axis)
            f = (1.0 / (x_max + 1e-9) ) [exp]
            
            self.fs += [f]
            self.fs = self.fs[-recalc_keep:]
            self.cur_f = sum(self.fs) / len(self.fs)  # Summary by average
        
            if 'batchnorm' in self.layer_type.lower():
                self.saved_x_max = x_max
        else:
            self.saved_x_max = None
            
        per_channel = 'batchnorm' in self.layer_type.lower()
        return self.cur_error_bound, self.cur_f, self.signed, per_channel
    
    @profile         
    def update_grads(self, grad_output=None, grad_input=None, grad_weight=None, grad_bias=None,
                     weight=None, save_var=None):
        recalc_keep = ACGC_QScheme.recalc_keep
        if not (self.iteration % config.recalc_interval == 0):
            self.saved_x_max = None
            self.iteration += 1
            return  # Not an update cycle
        
        layer = self.layer
        eps = self.eps
        
        is_layer = lambda *a: any(s in self.layer_type.lower() for s in a)
        
        if is_layer('conv', 'linear'):
            assert grad_output != None and grad_weight != None, 'Required args'
            
            gy = to_cupy(grad_output)
            gw = to_cupy(grad_weight)
            if is_layer('conv'):
                N,C,H,W = self.shape
                u = np.prod(layer.kernel_size) / np.prod(layer.stride)
            else:
                N,C = self.shape
                H,W = 1,1
                u = 1
            
            sgw2s = max((gw**2).sum(), 1e-16)
            tot_sy = max(u * (gy**2).sum(), 1e-16)
            assert not (tot_sy > 1e100).any(), 'Check for exploding gradients'
            
            numer = 2 * eps * sgw2s
            denom = N*C*H*W * tot_sy
            
            error_bound = cupy.sqrt(numer / denom).astype('f')[None, None, None, None]
            
            if config.debug_nan and cupy.isnan(error_bound).any():
                print('Nan Check failed!')
                pdb.set_trace()
            
        elif 'batchnorm' in self.layer_type.lower():
            assert weight != None and grad_weight != None and \
                grad_output != None and save_var != None and \
                grad_input != None, 'Required args'
            
            inv_std2 = save_var ** 2  # cudnn save_var is ACTUALLY 1/var (I think)
            gamma2 = weight ** 2
            ggamma2 = to_cupy(grad_weight) ** 2
            gx = to_cupy(grad_input)
            gy = to_cupy(grad_output)
            
            N,C,H,W = self.shape
            
            sgx2 = max(sum_of_squares(gx, axis=None), 1e-16)
            assert not (sgx2 > 1e100).any(), 'Check for exploding gradients'
            del gx
            max_dx = self.saved_x_max / 2   # TODO: Change, Dumb approx for now
            max_gy2 = (gy**2).max(axis=(0,2,3))
            nnz2 = cupy.count_nonzero(gy, axis=(0,2,3))**2
            
            approx_sum = inv_std2 * nnz2 * max_gy2 * (max_dx**2)
            
            numer = 2 * N*H*W * eps * sgx2
            denom = 2 * C * gamma2 * (inv_std2**2) * (approx_sum + ggamma2)
            denom = cupy.maximum(denom, 1e-16)
            error_bound = cupy.sqrt(numer / denom).astype('f')[None, :, None, None]
            
        else:
            raise Exception('Unrecognized layer {}'.format(self.layer_type))
        
        self.error_bounds += [error_bound]
        self.error_bounds = self.error_bounds[-recalc_keep:]
        self.cur_error_bound = sum(self.error_bounds) / len(self.error_bounds)  # Summary by average
        
        self.saved_x_max = None
        self.iteration += 1
        assert not cupy.isnan(error_bound).any(), 'Ensure error bound met'
        
        
    def if_allocate_perlayer(self):
        pass
        

sum_of_squares =  cupy.ReductionKernel(
        'T x', 'T y',
        'x * x',
        'a + b',
        'y = a',
        '0',
        'squared_l2norm_kernel')

        