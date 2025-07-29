import time
from contextlib import contextmanager
from contextlib import nullcontext
from rich.console import Console
from rich.table import Table
import torch
import subprocess
import torch

def gpu_mem_used(gpu_id: int = 0) -> int:
    """
    Retorna a memória usada em MiB reportada pelo nvidia-smi
    Para GPU única, use gpu_id=0.
    """
    cmd = [
        "nvidia-smi",
        f"--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
        f"--id={gpu_id}",
    ]
    out = subprocess.check_output(cmd, encoding="utf-8")
    return int(out.strip())   # MiB
class Profiler:
    def __init__(self, memory_profiler=False, gpu_id: int = 0):
        self.tempos = {}       # soma dos tempos
        self.chamadas = {}     # contador de chamadas
        self.mem_deltas = {}   # soma dos deltas de memória (bytes)
        self.memory_profiler = memory_profiler
        self.gpu_id = gpu_id
    def __call__(self, nome=None):
        @contextmanager
        def _medidor():
            t0 = time.time()
            if self.memory_profiler:
                m0 = gpu_mem_used(self.gpu_id)
            try:
                yield
            finally:
                td = time.time() - t0
                if self.memory_profiler:
                    torch.cuda.synchronize()  # garante que todas as operações CUDA terminaram
                    m1 = gpu_mem_used(self.gpu_id)
                    delta = m1 - m0
                # acumula tempo e chamadas
                self.tempos[nome] = self.tempos.get(nome, 0.0) + td
                self.chamadas[nome] = self.chamadas.get(nome, 0) + 1
                # acumula delta de memória
                if self.memory_profiler:
                    self.mem_deltas[nome] = self.mem_deltas.get(nome, 0) + delta
        return _medidor()
    def print_all(self):
        console = Console()
        table = Table(title=":bar_chart: Resultados do Profiler")
        table.add_column("Nome", style="bold cyan", no_wrap=True)
        table.add_column("Média (s)", justify="right", style="magenta")
        table.add_column("Total (s)", justify="right", style="green")
        table.add_column("Chamadas", justify="right", style="yellow")
        table.add_column("ΔMem (MB)", justify="right", style="red")
        table.add_column("Total ΔMem (MB)", justify="right", style="blue")
        for nome, total_t in self.tempos.items():
            # if nome.split('/')[0] not in ['decoding', 'prefill']:
            #     continue
            calls = self.chamadas[nome]
            avg_t = total_t / calls
            avg_delta_mb = (self.mem_deltas.get(nome, 0) / calls)
            total_delta_mb = self.mem_deltas.get(nome, 0)
            # coloriza nome como antes, se quiser
            nome_col = nome
            if "pretrain" in nome.lower():
                nome_col = f"[bold red]{nome}[/bold red]"
            elif "attention" in nome.lower():
                nome_col = f"[bold blue]{nome}[/bold blue]"
            elif "decoding" in nome.lower():
                nome_col = f"[bold yellow]{nome}[/bold yellow]"
            table.add_row(
                nome_col,
                f"{avg_t:.4f}",
                f"{total_t:.4f}",
                str(calls),
                f"{avg_delta_mb:.2f}",
                f"{total_delta_mb:.2f}",  # converte bytes para MB
            )
        console.print(table)

class SkipProfiler:
    def __call__(self, nome=None):
        return nullcontext()

def calcular_soma(medidor: Profiler | SkipProfiler = SkipProfiler()):
    with medidor("Exemplo de uma função que pode ou não receber o profiler"):
        soma = sum(i*i for i in range(100_000))
    return soma

# Exemplo de uso:
if __name__ == '__main__':
    profiler = Profiler(memory_profiler=True)
    with profiler('inicializacao'):
        torch.randn(1, 1, device='cuda')
    with profiler('processamento'):
        temp = torch.randn(1000, 1000, device='cuda')
    with profiler('processamento2'):
        temp2 = torch.randn(1000, 10000, device='cuda')
    profiler.print_all()