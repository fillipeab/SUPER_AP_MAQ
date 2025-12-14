import os

torchreid_path = os.path.dirname(__import__('torchreid').__file__)
rank_cylib_path = os.path.join(torchreid_path, 'reid', 'metrics', 'rank_cylib')

print("Arquivos na pasta rank_cylib:")
for item in os.listdir(rank_cylib_path):
    print(f"  {item}")

# Procura por arquivos .pyx
pyx_files = [f for f in os.listdir(rank_cylib_path) if f.endswith('.pyx')]
print(f"\nArquivos .pyx encontrados: {pyx_files}")
