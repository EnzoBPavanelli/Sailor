# Sailor – Sistema de Navegação Autônoma para Veleiros

Este repositório faz parte de um projeto de Iniciação Científica desenvolvido na Universidade Federal Fluminense (UFF), com foco em algoritmos de controle para veleiros autônomos. O sistema foi implementado em Python e está preparado para funcionar em conjunto com o simulador Gazebo e o framework ROS (Robot Operating System).

---

## Objetivo

Desenvolver e testar algoritmos de tomada de decisão para navegação de um veleiro autônomo, considerando:

- Obstáculos no ambiente (via sensores como LaserScan)
- Direção e intensidade do vento
- Estratégias heurísticas de controle

O foco principal deste repositório é a lógica de navegação e cálculo do ângulo ideal para seguir um caminho seguro e eficiente, respeitando limitações físicas de um veleiro (como não navegar contra o vento diretamente).

---

## Tecnologias e Ferramentas

- Python 3
- ROS (Robot Operating System)
- Gazebo (simulador 3D robótico)
- Heurísticas personalizadas para navegação
- Estrutura modular de controle

---

## Estrutura do Projeto

.
├── AnguloFinal.py          # Cálculo do ângulo final de navegação  
├── Angulos.py              # Gerador de possíveis ângulos de navegação  
├── apfvetorialvento.py     # Lógica de forças artificiais considerando vento  
├── calculateapf.py         # Função base de cálculo de força atrativa e repulsiva  
├── captain.py              # Classe principal de decisão e controle  
├── captain2.py             # Variante do capitão 

---

## Como Funciona

Os módulos implementam uma lógica baseada em Forças Potenciais Artificiais (APF) combinadas com heurísticas de navegação a vela. O sistema:

1. Gera um conjunto de possíveis ângulos de navegação.
2. Avalia cada ângulo com base em:
   - Proximidade de obstáculos (via LaserScan)
   - Direção do vento (wind heading)
   - Rumo anterior
3. Escolhe o ângulo que oferece o melhor equilíbrio entre segurança, eficiência e estabilidade.

Importante: Este repositório contém apenas a parte algorítmica do sistema. A integração com ROS e Gazebo foi realizada por outra parte da equipe e não está incluída aqui.

---

## Contexto Acadêmico

Projeto de Iniciação Científica – UFF  
Período: [2024–2025]  
Tema: Inteligência Computacional Aplicada à Navegação Autônoma de Veleiros

## Licença

Este projeto é de uso acadêmico e pessoal. Todos os direitos reservados ao autor.

---

## Contato

Enzo Pavanelli  
LinkedIn: https://www.linkedin.com/in/enzo-brito-pavanelli-269290256/
