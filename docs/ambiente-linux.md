# Ambiente inicial no Linux

O clone de trabalho fica em `~/EnsembleLearning` no servidor de testes. Os bancos são arquivos locais ignorados pelo Git; confira `data/README.md` e o contrato de dados antes de usá-los. Ative o ambiente com:

```bash
cd ~/EnsembleLearning
source .venv/bin/activate
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

O ambiente inicial usa Python 3.12, PyTorch 2.11.0 com CUDA 12.8, NumPy, pandas, SciPy e scikit-learn. O arquivo local `environment-linux.freeze.txt` registra as versões instaladas no servidor; ele não é uma especificação validada para todos os sistemas. O pacote PyTorch inclui bibliotecas CUDA para execução; o compilador `nvcc` e o toolkit de desenvolvimento completo não foram instalados. Caso futuros experimentos precisem compilar extensões, avalie e documente essa instalação separadamente.

A disponibilidade de CUDA não valida os dados ou o protocolo científico. Antes de treinar, confirme proveniência dos bancos, definição dos rótulos, partições e prevenção de vazamento conforme `docs/roadmap.md` e `docs/protocol.md`.
