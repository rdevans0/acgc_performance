#!/usr/bin/env python3

import numpy as np

import itertools

try:
    import cupy
    get_array_module = cupy.get_array_module
except ImportError:
    cupy = None
    get_array_module = lambda *a, **kw: np
    
try:
    import torch
except ImportError:
    torch = None

def _cupy_pack(x, bits, scale=None, signed=None):
    assert bits > 1 and bits <= 16
    
    if scale is None:
        scale = 1.0
    
    tmpl = '''
        uint16_t bits = {bits};
        unsigned signed_kernel = {signed};
        '''.format(bits=bits, signed=int(signed))
        
    
    quant_pack = cupy.ElementwiseKernel(
        'raw T x, raw int32 x_size, T scale', 'uint16 packed', 
        #'raw T x, raw int32 x_size, T scale', 'uint16 packed, uint16 s_bit, uint16 e_bit, uint16 k_rem, uint16 k, uint16 mask, uint16 value, raw uint16 p2', 
        tmpl + '''
            uint16_t mid, high;
            uint32_t s_bit, e_bit, k_rem;
            uint32_t k;
            uint16_t mask, value;
            T f; 
            
            mid = signed_kernel ? (1<<bits-1) : 0;
            high = (1<<bits);
            f = signed_kernel ? high*scale/2.0f : high*scale;
        
            s_bit = (i+1)*16 - 1; // start bit index      
            e_bit = i*16;         // bit index of i
            
            k = s_bit / bits;           // Index into x
            k_rem = s_bit + 1 - k*bits; // Number of bits remaining
            mask = 1 << ((bits - k_rem) % bits); // Bitmask
            mask = k < x_size ? mask : 0;
            
            T x_ = k < x_size ? x[k] : 0;
            T scaled = x_ * f + mid;
            value = (scaled < 0) ? 0 :
                    (scaled > (high-1)) ? (high-1) : 
                        __float2uint_rn(scaled);
                    
            
            for (unsigned j = 0; j < 16; j++) {
                uint32_t new_k = (s_bit - j) / bits;
                
                if (new_k != k) {  // Update value if moved to a new one
                    k = new_k;
                    mask = k < x_size ? 1 : 0;
                    T x_ = k < x_size ? x[k] : 0;
                    T scaled = x_ * f + mid;
                    
                    value = (scaled < 0) ? 0 :
                            (scaled > (high-1)) ? (high-1) : 
                                __float2uint_rn(scaled);
                }
                
                // Write bit
                packed |= (mask & value ? 1 : 0) << j ;
                //p2[i*16 + j] = value;
                mask = mask << 1;
                
            }
        ''',
        'quant_pack_{}bit_{}'.format(bits, ('s','u')[signed]),
        preamble='    typedef unsigned short uint16_t;\n     typedef unsigned uint32_t;\n')
    
    
    x = x.ravel()
    packed_size = (x.size*bits + 15) // 16
    packed = cupy.zeros((packed_size,), dtype=cupy.uint16)
    if 1:
        quant_pack(x, x.size, scale, packed)
    else:
        z = lambda : cupy.zeros((packed_size,), dtype=np.uint16)
        sbit, ebit, krem = z(), z(), z()
        k, mask, value = z(), z(), z()
        p2 = cupy.zeros((packed_size, 16), dtype=np.uint16)
        quant_pack(x, x.size, scale, packed, sbit, ebit, krem, k, mask, value, p2)
        
        print(f'{x=}')
        print(f'{packed=}')
        print(f'{sbit=}')
        print(f'{ebit=}')
        print(f'{krem=}')
        print(f'{k=}')
        print(f'{mask=}')
        print(f'{value=}')
        print('p2=\n' + np.array2string(p2.get(), max_line_width=128))
    
    
    return packed
    

def _cupy_unpack(packed, bits, scale=None, signed=None):
    assert bits > 1 and bits <= 16
    debug = False
    
    if scale is None:
        scale = 1.0
    
    
    tmpl = '''
        uint16_t bits = {bits};
        unsigned signed_kernel = {signed};
        '''.format(bits=bits, signed=int(signed))
    
    if not debug:
        tmpl += '''
            uint32_t s_bit, split_bit;
            uint16_t nl_bits, nr_bits;
            uint16_t left, right, unpacked; '''
        
        outputs = 'T x'
    else:
        outputs = 'T x, uint16 left, uint16 right, uint16 unpacked, uint16 s_bit, uint16 nl_bits, uint16 nr_bits, uint16 split_bit, uint16 tmp_out'
    
    quant_unpack = cupy.ElementwiseKernel(
        'raw uint16 packed, T scale', outputs,
        tmpl + '''
            uint16_t mid, high;
            uint32_t e_bit;
            
            s_bit = i*bits;    // start bit for the value we want
            e_bit = (i+1)*bits; // end bit for the value we want
            
            uint32_t k_left = s_bit / 16;
            left = packed[k_left];
            split_bit = (k_left+1) * 16;  // Split between this and the next value
            
            nl_bits = split_bit - s_bit;  // bits on left
            
            uint16_t mask_left = (1 << nl_bits) - 1; 
            
            if (split_bit >= e_bit) {
                // Value in one uint16
                nr_bits = split_bit - e_bit; // bits on right
                unpacked = 0;
                unpacked |= (left & mask_left) >> nr_bits; 
                
            } else {
                // Unpack from up to two ints of packed
                nr_bits = (e_bit - split_bit) % bits; // bits on right
                right = packed[k_left + 1];
                
                unpacked = 0;
                unpacked |= (left & mask_left) << nr_bits;
                unpacked |= right >> (16 - nr_bits);  // no need to mask, >> zero pads
                
            }
                
            // Unscale
            mid = signed_kernel ? (1<<bits-1) : 0;
            high = (1<<bits);
            T f = signed_kernel ? high*scale/2.0f : high*scale;
            
            T x_s = (T) unpacked;
            x = (x_s - mid) / f;
            
        ''',
        'quant_unpack_{}bit_{}'.format(bits, ('s','u')[signed]),
        preamble='    typedef unsigned short uint16_t;\n    typedef unsigned uint32_t;\n')
        
    size = (packed.size * 16) // bits
    
    x = cupy.zeros((size,), dtype=cupy.float32)
    if not debug:
        quant_unpack(packed, scale, x)
    else:
        z = lambda : cupy.zeros((size,), dtype=np.uint16)
        left, right, unpacked = z(), z(), z()
        s_bit, nl_bits, nr_bits, split_bit = z(), z(), z(), z()
        tmp_out = z()
        quant_unpack(packed, scale, x, left, right, unpacked, s_bit, nl_bits, nr_bits, split_bit, tmp_out)
        
        print(f'{x=}')
        print('packed=' + binary_repr(packed))
        print('left=' + binary_repr(left))
        print('right=' + binary_repr(right))
        print(f'{unpacked=}')
        print(f'{s_bit=}')
        print(f'{nl_bits=}')
        print(f'{nr_bits=}')
        print(f'{split_bit=}')
        
        print('tmp_out=' + binary_repr(tmp_out))
    
    
    return x


def _cupy_pack_1(x):
    x = x.ravel()
    packed_size = (x.size + 15) // 16
    
    quant_pack = cupy.ElementwiseKernel(
        'raw T x, raw int32 x_size', 'uint16 packed',
        '''for (int j = 0; j < 16; ++j) {
            int k = i * 16 + j;
            int bit = k < x_size && x[k] != 0;
            packed |= bit << (15 - j);
        }''','quant_pack_1')
        
    
    packed = cupy.zeros((packed_size,), dtype=cupy.uint16)
    quant_pack(x, x.size, packed)
    return packed

def _cupy_unpack_1(packed, scale=None, signed=True, signed_1bit=True):
    size = packed.size * 16
    
    signed_1bit = int(signed and signed_1bit)
    quant_unpack = cupy.ElementwiseKernel(
        'raw uint16 q, T scale', 'T unpacked',
        '''
        T lo = {signed_1bit} ? -1/scale : 0;
        T hi = 1 / scale;
        T b = (q[i / 16] >> (15 - i % 16)) & 1;
        unpacked = b ? lo : hi;
        '''.format(signed_1bit=signed_1bit),
        'quant_unpack_1_{}'.format(signed_1bit))
        
    
    x = cupy.zeros((size,), dtype=cupy.float32)
    
    quant_unpack(packed, scale, x)
    return x


def fixpoint_compress(x, bits, scale=None, signed=None, impl='pack'):
    xp = get_array_module(x)
    x =  x.view().flatten()
    dprint = lambda *a, **kw: None
    
    if signed is None:
        signed = x.min() < 0 # Note, this is an EXPENSIVE operation
    
    impl = impl.lower()
    
    if bits == 1 and not impl == 'cupy': # Common across all impls
        unpacked = (x >= 0).view().astype('uint8')
        q = xp.packbits(unpacked)
    elif bits == 1 and impl == 'cupy':
        q = _cupy_pack_1(x)
    
    elif impl == 'cupy':
        q = _cupy_pack(x, bits, scale=scale, signed=signed)
    
    elif impl == 'pack':
        # Method going from uint16 -> unpack -> trim(uintX) -> pack
        dtype, dbits = np.uint8, 8
        assert dbits >= bits
        
        mid = (1<<bits-1)
        high = (1<<bits)
        sf = high if scale is None else high*scale
        
        if not signed:
            x_s = x * sf                  # [0, hi]
        else:
            x_s = x * (sf / 2.0) + mid    # [0, hi]
        
        xp.rint(x_s, out=x_s)  # Round to nearest int
        xp.clip(x_s, 0, high - 1, out=x_s)  # Truncate high
        
        dprint('x_s', x_s)
        
        q_c = x_s.astype(dtype)
        dprint('q_c', q_c)
        
        q_u = xp.unpackbits(q_c).reshape(-1, dbits)
        dprint('q_u', q_u)
        
        # Trim and repack
        q_t = q_u[:, -bits:]
        q = xp.packbits(q_t.flatten())
        
    
    else:
        raise NotImplementedError('Unknown impl {}'.format(impl))
    
    return q


def fixpoint_decompress(quant, bits, scale=None, signed=None, signed_1bit=True, impl='pack', dtype=np.float32):
    xp = get_array_module(quant)
    quant =  quant.view().flatten()
    dprint = lambda *a, **kw: None
    
    assert signed is not None, 'Need a sign as input'
    
    impl = impl.lower()
    sf = xp.array(1, dtype=dtype) if scale is None else scale.astype(dtype)
    
    if bits == 1 and impl == 'cupy':
        x = _cupy_unpack_1(quant, sf, signed=signed, signed_1bit=signed_1bit)
        
    elif bits == 1 and signed and signed_1bit: 
        unpacked = xp.unpackbits(quant).flatten()
        x = xp.where(unpacked, 1/sf, -1/sf).astype(dtype)  # 1-> 1, 0-> -1
        
    elif bits == 1:
        unpacked = xp.unpackbits(quant).flatten()
        x = xp.where(unpacked, 1/sf, 0/sf).astype(dtype)
        
    elif impl == 'cupy':
        x = _cupy_unpack(quant, bits, scale=sf, signed=signed)
    
    elif impl == 'pack':
        idbits = 8
        # Method going from uint16 -> unpack -> trim(uintX) -> pack
        assert idbits >= bits
        
        unpacked = xp.unpackbits(quant).reshape(-1, bits)
        
        dprint('unpacked',unpacked)
        
        # Pad up to d bits
        pw = idbits - bits
        if pw == 0:
            q_u = unpacked
        else:
            q_u = xp.pad(unpacked, [(0,0),(pw,0)], 
                            mode='constant', constant_values=0)
            del unpacked
            
        dprint('q_u', q_u)
        
        x_s = xp.packbits(q_u.flatten()).astype(dtype)
        
        dprint('x_s', x_s)
        
        mid = (1<<bits-1)
        high = (1<<bits)
        
        if not signed:
            sf2 = sf * high
            x = x_s / sf2   # [0, 1/sf]
        else:
            sf2 = sf * high / 2.0
            x = (x_s - mid) / sf2   # [-1/sf, 1/sf]
            
    
    else:
        raise NotImplementedError('Unknown impl {}'.format(impl))
        
    return x

###############################################################################
###############################################################################

def split_u16(x):
    if x.dtype == np.uint8:
        return x.copy()
    x = x.flatten()
    xp = get_array_module(x)
    x_l = (x & 0x00FF) >> 0
    x_h = (x & 0xFF00) >> 8
    return xp.concatenate((x_h[:,None], x_l[:,None]), axis=1).flatten()

def join_u8(x):
    if x.dtype == np.uint16:
        return x.copy()
    xp = get_array_module(x)
    
    x_ = x.flatten().reshape(x.size//2,2).astype(np.uint16)
    return xp.bitwise_or(x_[:,0] << 8, x_[:,1])
    
    
    
    
def array2string(x):
    xp = get_array_module(x)
    return np.array2string(xp.around(x, 3).get(), max_line_width=255, formatter={'all':'{:4.2f}'.format}, separator=', ')
    
    

def binary_repr(x, bits=None):
    if get_array_module(x) != np:
        x = cupy.asnumpy(x)
    
    # if x.dtype.itemsize == 2:
    #     x_l = (x & 0x00FF) >> 0
    #     x_h = (x & 0xFF00) >> 8
    #     x = np.concatenate((x_h[:,None], x_l[:,None]), axis=1).flatten()
    # bits = 8
    
    bits_ = x.itemsize * 8
    if bits is None:
        return repr([np.binary_repr(v, width=bits_) for v in x.flatten()])
    else:
        all_bits = ''.join(np.binary_repr(v, width=bits_) for v in x.flatten())
        numel = len(all_bits) // bits
        items = [all_bits[i*bits:(i+1)*bits] for i in range(numel)]
        return repr(items)

def sizeof(x):
    return x.size * x.itemsize

#@profile
def time_single():
    xp = cupy
    shape = (128, 1024, 8, 8)
    N,C,H,W = shape
    
    signed = True
    bits = 1
    seed = 0x0f973ab
    xp.random.seed(seed)
    impl = 'cupy'
    
    
    x = (xp.random.random(shape) - 0.5)
    
    
    import time
    
    start = time.time()
    
    quant = []
    xt = x.transpose(1,0,2,3).reshape(C, -1)
    cupy.cuda.Device(0).synchronize()
    for xc in xt:
        qc = fixpoint_compress(xc, bits, signed=signed, impl=impl)
        cupy.cuda.Device(0).synchronize()
        quant.append(qc)
    
    xt_ = xp.empty_like(xt)
    for c, qc in enumerate(quant):
        xt_[c,...] = fixpoint_decompress(qc, bits, signed=signed, impl=impl)
        cupy.cuda.Device(0).synchronize()
    
    x_ = xt_.reshape(C,N,H,W).transpose(1,0,2,3).reshape(shape)
    cupy.cuda.Device(0).synchronize()
    
    end = time.time()
    
    t = end-start
    print('Completed in {:.2f}s ({:.2f}ms/channel, {:.2f}/s)'.format(t, t*1000/C, 1/t))

def test_single():
    xp = cupy
    
    if 0:
        shape=(32,)
        sparsity = 0.5
        seed = 0x0f973ab
        bits = 12
        
        xp.random.seed(seed)
        
        #m = (xp.random.random(shape) > sparsity)
        #x = m * (xp.random.random(shape) - 0.5)
        
        x = xp.zeros(shape, dtype='f')
        
        x[:5] = xp.array([-2, -1, 0, 1, 2])
        x[5:] = xp.linspace(-1.3, 1.3, num=x.size-5)
    if 1:
        x = xp.asarray(np.load('1_x_pre_c.npz')['x_f']).flatten()
        x = x#[3000:6000]
        bits = 12
        
    signed = bool(x.min() < 0)
    
    print('Bits {}'.format(bits))
    print('Signed {}'.format(signed))
    print('Shape {}'.format(x.shape))
    
    
    quant = fixpoint_compress(x, bits, signed=signed, impl='cupy')
    
    x_r = fixpoint_decompress(quant, bits, signed=signed, impl='cupy')
    x_ = x_r.reshape(x.shape)
    
    
    
    osize = sizeof(x)
    csize = sizeof(quant)
    print('{} -> {} ({}x)'.format(osize, csize, osize/csize))
    
    max_err = 2**-bits if not signed else 2**(-bits+1)
    err = abs(x - x_)[(x >= -1) * (x <= 1)].max()
    print('Actual {}   Max {}'.format(err, max_err))
    
    
    max_levels = 2**bits
    _, counts = xp.unique(x_, return_counts=1)
    act_levels = counts.size
    print('Levels {}  Max {}'.format(act_levels, max_levels))
    
    ctx = '\n'
    ctx += '\n' + array2string(x)
    ctx += '\n\n' + binary_repr(quant)
    ctx += '\n\n' + binary_repr(quant, bits=bits)
    ctx += '\n\n' + array2string(x_)
    ctx += '\n' + '-'*32 + '\n'
    
    if err > 1.01 * max_err:
        
        print('\n' + '-'*32 + '\n')
        print('Error too high!!!')
    
        print(ctx)    
            
        m = (x >= -1) * (x <= 1)
        print(xp.around(x[m],3))
        print(xp.around(x_[m],3))
        print(xp.around(abs(x - x_)[m],3))
        print((abs(x - x_)[m] > 1.01 * max_err)*1)
        mask = abs(x - x_) > (1.01 * max_err) * m
        raise Exception('Error too high!')
    
    if err.max() <= max_err:
        print('Output is SAME')
    else:
        print('Output is DIFFERENT')

def check_same():
    xp = cupy
    shape=(32,)
    
    for signed in (True, False):
            x = xp.zeros(shape, dtype='f')
            if signed:
                x[:5] = xp.array([-2, -1, 0, 1, 2])
                x[5:] = xp.linspace(-1.3, 1.3, num=x.size-5)
            else:
                x[:5] = xp.array([0,0.25, 0.5, 1, 2])
                x[5:] = xp.linspace(0, 1.3, num=x.size-5)
                
            for bits in range(2,9):
                q1 = fixpoint_compress(x, bits, signed=signed, impl='cupy')
                q2 = fixpoint_compress(x, bits, signed=signed, impl='pack')
                
                s1 = binary_repr(q1)
                s2 = binary_repr(join_u8(q2))
                
                same = [s1[k] == s2[k] for k in range(len(s1))]
                
                if not all(same):
                    diff = ['^ '[s] for s in same]
                    print('cupy ' + s1)
                    print('pack ' + s2)
                    print('diff ' + ''.join(diff))
                    
                
                    raise Exception('Implementation diff found!')
    
    return

def test_perf(N=5, print=print):
    sizes_log2 = (14, 18, 24) 
    signed = True
    methods = ('cupy','pack')
    bitwidths = (1,2,3,4,6,7,8)
    xp = cupy
    
    
    import time
    # def time_call(func, *a, **kw):
    #     start = time.time()
    #     ret = func(*a, **kw)
    #     end = time.time()
    #     time_ms = (end - start) * 1000
    #     return (time_ms, ret)
    
    def time_cd(N=5, get_x=None, **kw):
        tc_avg = 0
        td_avg = 0
        xs = [get_x() for _ in range(N)]
        if xp == cupy:
            cupy.cuda.stream.get_current_stream().synchronize()
        
        qs = [None for _ in range(N)]
        ret = [None for _ in range(N)]
        
        # Do Compression and decompression in blocks to avoid launch overheads
        
        start = time.time()
        for n in range(N):
            qs[n] = fixpoint_compress(xs[n], bits, **kw)
        if xp == cupy:
            cupy.cuda.stream.get_current_stream().synchronize()
        end = time.time()
        
        tc_avg = (end - start) * 1000 / N
        
        
        start = time.time()
        for n in range(N):
            ret[n] = fixpoint_decompress(qs[n], bits, **kw)
        if xp == cupy:
            cupy.cuda.stream.get_current_stream().synchronize()
        end = time.time()
        
        td_avg = (end - start) * 1000 / N
        
        return tc_avg, td_avg
    
    
    fmt = ('{size:>10} {bits:>8} '
           + ' '.join('{c_ms[%s]:>20}'%(m) for m in methods) + ' '
           + ' '.join('{d_ms[%s]:>20}'%(m) for m in methods)
           ).format
        
    
    heading = fmt(size='Size (log2)', bits='Bits', 
                  c_ms={m:'%s compr ms'%m for m in methods},
                  d_ms={m:'%s cecompr ms'%m for m in methods},
                  )
    print(heading)
    
        
    
    for size in sizes_log2:
        if signed:
            get_x = lambda : (xp.random.random(size=2**size, dtype='f') - 0.5)*2.4
        else:
            get_x = lambda : xp.random.random(size=2**size, dtype='f') * 1.2
        
        for bits in bitwidths:
            c_ms = {}
            d_ms = {}
            
            for method in methods:
                tc, td = time_cd(N=N, get_x=get_x, scale=None, signed=signed, 
                                 impl=method)
                
                c_ms[method] = tc
                d_ms[method] = td
            
            best_c = min(c_ms.values())
            best_d = min(d_ms.values())
            fmt_best = lambda d,m,b : '{}{:.3f}'.format('*** ' if d[m]==b else '', d[m])
            for m in methods:
                c_ms[m] = fmt_best(c_ms, m, best_c)
                d_ms[m] = fmt_best(d_ms, m, best_d)
                
            print(fmt(**vars()))
            
            
        
        
        
        

if __name__ == '__main__':
    #check_same()
    #time_single()
    #test_single()
    test_perf()
