#!/usr/bin/env python3

import numpy as np

try:
    import cupy
    get_array_module = cupy.get_array_module
except ImportError:
    get_array_module = lambda *a, **kw: np



def zvc_compress(x, impl='mask'):
    xp = get_array_module(x)
    
    x =  x.view().flatten()
    
    impl = impl.lower()
    if impl == 'mask':
        mask = (x != 0)  # Locate nonzeros
        values = x[mask]  # Extract nonzeros
        nonzeros = xp.packbits(mask).flatten()  # Pack into uint8
        
    elif impl == 'take':
        inds = xp.argwhere(x).flatten() # Locate nonzeros
        values = x.take(inds)  # Extract nonzeros
        mask =  (x != 0)
        nonzeros = xp.packbits(mask).flatten()  # Pack into uint8
    else:
        raise NotImplementedError('Unknown impl {}'.format(impl))
    
    return values, nonzeros

def zvc_decompress(vals, nzvs, shape, impl='mask'):
    xp = get_array_module(vals)
    
    impl = impl.lower()
    numel = np.prod(shape)
    multiple_of_8 = (numel % 8) == 0
    if impl in ('mask', 'take'):
        mask = xp.unpackbits(nzvs)
        if not multiple_of_8:
            mask = mask[:numel]
        mask = mask.view().reshape(shape).astype('bool')
        
        x = xp.zeros(shape, dtype=vals.dtype)
        x[mask] = vals
        
    else:
        raise NotImplementedError('Unknown impl {}'.format(impl))
    
    return x
    
    

###############################################################################
###############################################################################

def binary_repr(x):
    return repr([np.binary_repr(v, width=8) for v in x.flatten()])

def sizeof(x):
    return x.size * x.itemsize

def test_single():
    xp = np
    shape=(64,)
    sparsity = 0.5
    seed = 0x0f973ab
    impl = 'mask'
    
    xp.random.seed(seed)
    
    m = (xp.random.random(shape) > sparsity)
    x = m * xp.random.random(shape) * xp.random.randint(100)
    
    print(x)
    print('-'*32)
    print('')
    
    vals, nzvs = zvc_compress(x, impl='take')
    
    print(vals)
    print(binary_repr(nzvs))
    print('-'*32)
    print('')
                
    x_ = zvc_decompress(vals, nzvs, shape, impl=impl)
    
    print(x_)
    print('')
    
    osize = sizeof(x)
    csize = sizeof(vals) + sizeof(nzvs)
    print('{} -> {} ({}x)'.format(osize, csize, osize/csize))
    
    if (x == x_).all():
        print('Output is SAME')
    else:
        print('Output is DIFFERENT')



def test(num=10, seed=0x0f973ab):
    impls = ['mask', 'take']
    xps = [np]
    shapes = [(100,), (128,3,4,4), (64,), (5,3)]
    sparsities = (0,1,10,50,90,99,100)
    
    for i, xp in enumerate(xps):
        xp.random.seed(seed ^ i)
        
        
    for sparsity in sparsities:
        for shape in shapes:
            for xp in xps:
                mask = (xp.random.random(shape) > (sparsity/100))
                x = mask * xp.random.random(shape) * xp.random.randint(100)
                
                for impl in impls:
                    print('{}%, {}, "{}"'.format(sparsity, shape, impl))
                    vals, nzvs = zvc_compress(x, impl=impl)
                    x_ = zvc_decompress(vals, nzvs, shape, impl=impl)
                    
                    if not (x_ == x).all():
                        print('    FAIL')
                        raise Exception('FAILED!')
                    else:
                        print('    PASS')

if __name__ == '__main__':
    #test_single()
    test()