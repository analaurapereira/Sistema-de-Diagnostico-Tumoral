import os
os.environ['QT_LOGGING_RULES'] = '*=false'
import cv2
import numpy as np
import kagglehub
import random

print("=== SISTEMA DE DIAGNÓSTICO ===")

while True:
    sexo = input("Digite o sexo biológico do paciente (M/F): ").strip()
    if sexo.upper() in ('M', 'F'):
        break
    print("  [ERRO] Valor inválido. Digite apenas 'M' para masculino ou 'F' para feminino.")

# Validação da altura
while True:
    altura = input("Digite a altura do paciente (cm): ").strip()
    try:
        alt_val = float(altura)
        if alt_val <= 0:
            print("  [ERRO] A altura deve ser um número positivo.")
        else:
            break
    except ValueError:
        print("  [ERRO] Valor inválido. Digite um número (ex: 175).")

# Baixa a base de dados de imagens de ressonância (Kaggle)
path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")

# Escolhe aleatoriamente uma imagem com ou sem tumor
pasta_escolhida = random.choice(['yes', 'no'])
subpasta = os.path.join(path, pasta_escolhida)

img_name = random.choice([f for f in os.listdir(subpasta) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
img_path = os.path.join(subpasta, img_name)

# Lê a imagem e cria uma cópia para desenhar o resultado
img = cv2.imread(img_path)
diagnostico = img.copy()

# Converte a imagem colorida para Cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplica um desfoque  para remover pequenos artefatos
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Aplica limite baixo de cor para enxergar apenas a cabeça toda
_, thresh_cerebro = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
kernel = np.ones((5,5), np.uint8)

# Limpa sujeiras morfológicas da imagem
thresh_cerebro = cv2.morphologyEx(thresh_cerebro, cv2.MORPH_OPEN, kernel, iterations=1)

# Traça o contorno do cérebro inteiro
contours_cerebro, _ = cv2.findContours(thresh_cerebro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

area_cerebro = 0
if contours_cerebro:
    c_cerebro_max = max(contours_cerebro, key=cv2.contourArea)
    area_cerebro = cv2.contourArea(c_cerebro_max)

# Aplica um limite alto para enxergar só partes muito brancas (anomalias)
_, thresh = cv2.threshold(blur, 155, 255, cv2.THRESH_BINARY)

# Preenche buracos e limpa pontos fora da anomalia principal
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Extrai o contorno da massa tumoral
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

tumor_encontrado = False
area_suspeita = 0

if contours:
    # Seleciona o maior contorno branco achado
    c_max = max(contours, key=cv2.contourArea)
    area_suspeita = cv2.contourArea(c_max)
    
    # Verifica se a área suspeita é maior que 300 pixels
    if area_suspeita > 300:
        tumor_encontrado = True
        
        # Desenha a linha vermelha em volta da área suspeita
        cv2.drawContours(diagnostico, [c_max], -1, (0, 0, 255), 2)
        
# Utilizando a fórmula baseada em altura e sexo (Ho et al.)
peso_cerebro_g = 0

if sexo.upper() == 'M':
    peso_cerebro_g = 920 + 2.70 * alt_val
else:
    peso_cerebro_g = 748 + 3.10 * alt_val


print("--- Relatório de Scanner MRI (Cérebro) ---")
print(f"Paciente: Sexo {sexo.upper()} | Altura: {altura}cm")
print(f"Arquivo analisado: {img_name}")
print(f"Diagnóstico real do dataset: {pasta_escolhida.upper()}")

if tumor_encontrado and area_cerebro > 0:
    pct = (area_suspeita / area_cerebro) * 100
    if pct > 1.0:
        print(f"\n[ALERTA]: Possível massa tumoral detectada pelo algoritmo!")
        print(f">> Porcentagem da massa afetada pela anomalia: {pct:.2f}%")

        if peso_cerebro_g > 0:
            peso_tumor_g = peso_cerebro_g * (pct / 100)
            print(f">> Peso estimado do cérebro: {peso_cerebro_g:.2f}g")
            print(f">> Peso estimado da área afetada: {peso_tumor_g:.2f}g")
    else:
        print("\n[NORMAL]: Nenhuma anomalia significativa foi detectada nos testes primários.")
else:
    print("\n[NORMAL]: Nenhuma anomalia significativa foi detectada nos testes primários.")

# Redimensiona as imagens para exibição
LARGURA_EXIBICAO = 200
def redimensionar(imagem):
    h, w = imagem.shape[:2]
    nova_altura = int(h * LARGURA_EXIBICAO / w)
    return cv2.resize(imagem, (LARGURA_EXIBICAO, nova_altura))

# Abre na tela as 4 fases do processamento
cv2.imshow('MRI Original', redimensionar(img))
cv2.imshow('Massa Craniana', redimensionar(thresh_cerebro))
cv2.imshow('Processamento(Threshold)', redimensionar(thresh))
cv2.imshow('Diagnostico', redimensionar(diagnostico))

print("\n(Pressione qualquer tecla nas janelas de imagem para fechar o programa)")
cv2.waitKey(0)
cv2.destroyAllWindows()
