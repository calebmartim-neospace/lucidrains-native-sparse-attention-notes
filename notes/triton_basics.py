## BEGIN helpers
import os
from IPython.core.debugger import set_trace

os.environ['TRITON_INTERPRET'] = '0' # needs to be set *before* triton is imported

def check_tensors_gpu_ready(*tensors):
    for t in tensors:
        # BEGIN Minha adição
        if not t.is_contiguous():
                t = t.contiguous()
        # END Minha adição
        assert t.is_contiguous, "A tensor is not contiguous"
        if not os.environ.get('TRITON_INTERPRET') == '1': assert t.is_cuda, "A tensor is not on cuda"

def test_pid_conds(conds, pid_0=[0], pid_1=[0], pid_2=[0]):
    '''Test if condition on pids are fulfilled
    E.g.:
        '=0'  checks that pid_0 == 0
        ',>1' checks that pid_1 > 1
        '>1,=0' checks that pid_0 > 1 and pid_1 == 0
    '''
    pids = pid_0[0], pid_1[0], pid_2[0]
    conds = conds.replace(' ','').split(',')
    for i, (cond, pid) in enumerate(zip(conds, pids)):
        if cond=='': continue
        op, threshold = cond[0], int(cond[1:])
        if op not in ['<','>','>=','<=','=', '!=']: raise ValueError(f"Rules may only use these ops: '<','>','>=','<=','=', '!='.")
        op = '==' if op == '=' else op
        if not eval(f'{pid} {op} {threshold}'): return False
    return True

assert test_pid_conds('')
assert test_pid_conds('>0', [1], [1])
assert not test_pid_conds('>0', [0], [1])
assert test_pid_conds('=0,=1', [0], [1], [0])

def breakpoint_if(conds, pid_0=[0], pid_1=[0], pid_2=[0]):
    '''Stop kernel, if any condition of pids is fulfilled'''
    if test_pid_conds(conds, pid_0, pid_1, pid_2): set_trace()

def print_if(txt, conds, pid_0=[0], pid_1=[0], pid_2=[0]):
    '''Print txt, if any condition of pids is fulfilled'''
    if test_pid_conds(conds, pid_0, pid_1, pid_2): print(txt)

# Teto da divisão:
def cdiv(a:int,b:int) -> int: return (a + b - 1) // b
## END helpers

import torch
from torch import tensor
import triton
import triton.language as tl

## Teste 1: copiando um tensor

def copy(x:torch.tensor, bs:int, kernel_fn) -> tensor:
    # Copia a forma e tipo de x em z, com todos os valores em 0:
    z:torch.tensor = torch.zeros_like(x)
    # Verifica se os elementos dos tensors estão guardados em uma sequência contígua da memória:
    check_tensors_gpu_ready(x, z)
    # Número de elementos totais em x:
    n:int = x.numel()
    # Quantos blocos de threads serão necessários para processar cada um dos n elementos:
    n_blocks = cdiv(n, bs)
    # Configuração de dimensões do kernel (chamado de 'program', em triton), pode ser 
    # uma tupla 1d, 2d ou 3d.
    grid = (n_blocks,)
    kernel_fn[grid](x, z, n, bs)

    return z

@triton.jit
def copy_k(x_ptr:int, z_ptr:int, n:int, bs:tl.constexpr) -> None:
    # Qual thread é a atual na dimensão 0 do grid:
    pid = tl.program_id(0)
    # seja y = (pid * bs), y é quantas threads estão atrás da atual,
    # offs = [y + 0, y + 1, y + 2, ... y + (bs - 1)]:
    offs = pid * bs + tl.arange(0, bs) 
    # mask[i] = True, se offs[i] < n,
    # mask[i] = False, caso contrário,
    # len(mask) = bs:
    mask = offs < n
    # Insere os elementos nos endereços [x_ptr + offs[0], x_ptr + offs[1], ..., x_ptr + offs[bs - 1]] em 
    # um vetor x, para todo índice i que atenda mask[i] = (offs[i] < n):
    x = tl.load(x_ptr + offs, mask)
    # Insere os elementos de x nos endereços [z_ptr + offs[0], z_ptr + offs[1], ... z_ptr + offs[bs - 1]]
    # seguindo as mesmas condições acima:
    tl.store(z_ptr + offs, x, mask)


x:tensor = tensor([1, 2, 3, 4], device='cuda')

z:tensor = copy(x, bs = 2, kernel_fn = copy_k)

# Output: tensor x: tensor([1, 2, 3, 4])
#         tensor z: tensor([1, 2, 3, 4])
print(f'tensor x: {x}\ntensor z: {z}')

# Teste 2: kernel 2d

def kernel_2d(h:int, w:int, bs0:tl.constexpr, bs1:tl.constexpr):
    # Linha da thread atual no grid: 
    pid_0 = tl.program_id(0)
    # Coluna da thread atual no grid: 
    pid_1 = tl.program_id(1)

    offs_0 = bs0 * pid_0 + tl.arange(0, bs0)
    offs_1 = bs1 * pid_1 + tl.arange(0, bs1)

    # cada elemento se aprofunda para uma dimensão
    # [a, b, c] -> [[a], [b], [c]]
    # equivalente a offs_0 = offs_0[:,None]
    offs_0 = tl.expand_dims(offs_0, 1) 
    # cada elemento se junta numa dimensão anterior
    # [a, b, c] -> [[a, b, c]]
    # equivalente a offs_1 = offs_1[None,:]
    offs_1 = tl.expand_dims(offs_1, 0) 

    # (w * offs_0) conta quantos elementos já passaram acima
    # dessas linhas e somar offs_1 introduz cada colunas (bs1 colunas) 
    # e soma como offsets para offs seja uma matriz 2d que possui todos
    # os índices que serão cobertos nesse program:
    offs = w * offs_0 + offs_1


# Teste 3: Multiplicação de matrizes

@triton.jit
def get_1d_offset(block_size:int, cur_pid:int) -> tensor:
    return cur_pid * block_size + tl.arange(0, block_size)

@triton.jit
def get_2d_offset(offs_0:int, offs_1:int, num_columns:int) -> tensor:
    return tl.expand_dims(offs_0, 1) * num_columns  + tl.expand_dims(offs_1, 0) 

@triton.jit
def get_1d_mask(offs, max) -> tensor:
    return offs < max

@triton.jit
def get_2d_mask(offs_0, offs_1, max_0, max_1):
    return (tl.expand_dims(offs_0, 1) < max_0) & (tl.expand_dims(offs_1, 0) < max_1)

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, m, n, k, bs:tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = get_1d_offset(bs, pid_m)
    offs_n = get_1d_offset(bs, pid_n)

    # Caso especial, a gente finge que k tem uma dimensão no bloco para podermos criar o
    # offset 2d:
    offs_k = get_1d_offset(bs, 0)

    indexes_a = a_ptr + get_2d_offset(offs_m, offs_k, num_columns = k)
    indexes_b = b_ptr + get_2d_offset(offs_k, offs_n, num_columns = n)

    acc = tl.zeros((bs, bs), dtype=tl.float32)
    
    for _ in range(0, k, bs):
        a = tl.load(indexes_a)
        b = tl.load(indexes_b)

        acc += tl.dot(a, b, allow_tf32=False)

        indexes_a += bs
        indexes_b += bs * n

    indexes_c = c_ptr + get_2d_offset(offs_m, offs_n, num_columns = n)
    mask = get_2d_mask(offs_m, offs_n, m, n)
    tl.store(indexes_c, acc, mask=mask) 

def matmul(a, b, bs):
    assert a.shape[1] == b.shape[0], "Linha precisa ter o mesmo tamanho de colunas"
    check_tensors_gpu_ready(a, b)
    m, n, k = a.shape[0], b.shape[0], a.shape[1]
    c = torch.empty((m, n), device='cuda', dtype=torch.float16)
    grid = (cdiv(m, bs), cdiv(n, bs))
    matmul_kernel[grid](a, b, c, m, n, k, bs)
    return c

# Teste pequeno:
a = tensor([[1, 2], [3, 4]], dtype=torch.float32, device = 'cuda')
b = tensor([[5, 6], [7, 8]], dtype=torch.float32, device = 'cuda')

# Output : [[19, 22],
#           [43, 50]]
print(matmul(a, b, 16))

# Teste grande: 
torch.manual_seed(0)

a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16)

triton_output = matmul(a, b, 16)
torch_output = torch.matmul(a, b)

print(triton_output)
print(torch_output)

if torch.allclose(triton_output, torch_output, atol=5e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")