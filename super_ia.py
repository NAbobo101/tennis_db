# -*- coding: utf-8 -*-
"""
Simulador de Previsão de Partidas de Tênis (ATP/WTA)
Versão Especialista com Features Completas do Match Charting Project
"""

# === IMPORTAÇÕES ===
import os
import warnings
import numpy as np
import pandas as pd
from collections import deque
from thefuzz import process
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.simplefilter(action='ignore', category=FutureWarning)

# === FUNÇÃO PRINCIPAL DE TREINAMENTO ===

def carregar_e_treinar(prefixo_liga):
    """
    Carrega os dados das partidas ATP/WTA e treina o modelo usando estatísticas básicas.
    """
    print(f"\n=== INICIANDO PROCESSAMENTO DA LIGA: {prefixo_liga.upper()} ===")
    
    pasta_base = 'database'
    anos = range(2000, 2024)  # Usando dados de 2000-2023
    
    # Carregamento dos dados
    frames = []
    for ano in anos:
        arquivo = os.path.join(pasta_base, f'{prefixo_liga}_matches_{ano}.csv')
        if os.path.exists(arquivo):
            try:
                df = pd.read_csv(arquivo)
                frames.append(df)
            except Exception as e:
                print(f"[AVISO] Erro ao carregar {arquivo}: {e}")
                continue
    
    if not frames:
        print("[ERRO] Nenhum arquivo de dados encontrado.")
        return None, None, None
        
    partidas = pd.concat(frames, ignore_index=True)
    
    # Seleção e preparação das features
    colunas_necessarias = ['winner_name', 'loser_name', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 
                          'w_1stWon', 'w_2ndWon', 'w_bpSaved', 'l_ace', 'l_df', 'l_svpt', 
                          'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_bpSaved']
    
    # Removendo linhas com valores ausentes nas colunas necessárias
    partidas = partidas.dropna(subset=colunas_necessarias)
    
    # Criando features normalizadas
    historico_features = {}
    features_para_modelo = []
    
    for _, partida in partidas.iterrows():
        vencedor, perdedor = partida['winner_name'], partida['loser_name']
        
        # Calculando métricas normalizadas para o vencedor
        w_stats = {
            'serve_pts_won_pct': (partida['w_1stWon'] + partida['w_2ndWon']) / max(partida['w_svpt'], 1),
            'ace_pct': partida['w_ace'] / max(partida['w_svpt'], 1),
            'df_pct': partida['w_df'] / max(partida['w_svpt'], 1),
            'first_serve_in_pct': partida['w_1stIn'] / max(partida['w_svpt'], 1),
            'first_serve_win_pct': partida['w_1stWon'] / max(partida['w_1stIn'], 1),
            'second_serve_win_pct': partida['w_2ndWon'] / max(partida['w_svpt'] - partida['w_1stIn'], 1),
            'bp_saved_pct': partida['w_bpSaved'] / max(partida['w_bpSaved'], 1),
            'bp_converted_pct': 0  # Placeholder para manter compatibilidade
        }
        
        # Calculando métricas normalizadas para o perdedor
        l_stats = {
            'serve_pts_won_pct': (partida['l_1stWon'] + partida['l_2ndWon']) / max(partida['l_svpt'], 1),
            'ace_pct': partida['l_ace'] / max(partida['l_svpt'], 1),
            'df_pct': partida['l_df'] / max(partida['l_svpt'], 1),
            'first_serve_in_pct': partida['l_1stIn'] / max(partida['l_svpt'], 1),
            'first_serve_win_pct': partida['l_1stWon'] / max(partida['l_1stIn'], 1),
            'second_serve_win_pct': partida['l_2ndWon'] / max(partida['l_svpt'] - partida['l_1stIn'], 1),
            'bp_saved_pct': partida['l_bpSaved'] / max(partida['l_bpSaved'], 1),
            'bp_converted_pct': 0  # Placeholder para manter compatibilidade
        }
        
        # Atualizando histórico dos jogadores
        hist_vencedor = historico_features.setdefault(vencedor, deque(maxlen=30))
        hist_perdedor = historico_features.setdefault(perdedor, deque(maxlen=30))
        
        hist_vencedor.append(w_stats)
        hist_perdedor.append(l_stats)
        
        # Calculando features para o modelo (diferenças entre médias móveis)
        avg_vencedor = pd.DataFrame(list(hist_vencedor)).mean()
        avg_perdedor = pd.DataFrame(list(hist_perdedor)).mean()
        
        features = {}
        for col in w_stats.keys():
            features[f'diff_{col}'] = avg_vencedor.get(col, 0) - avg_perdedor.get(col, 0)
            
        features_para_modelo.append(features)
    
    # Preparação e treinamento do modelo
    print("[INFO] Treinando modelo XGBoost...")
    df = pd.DataFrame(features_para_modelo).fillna(0)
    
    X = df
    y = pd.Series([1] * len(df))
    X_inv = -X
    y_inv = pd.Series([0] * len(df))
    
    X_full = pd.concat([X, X_inv], ignore_index=True)
    y_full = pd.concat([y, y_inv], ignore_index=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, stratify=y_full, random_state=42
    )
    
    modelo = XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        use_label_encoder=False,
        random_state=42
    )
    
    modelo.fit(X_train, y_train, verbose=False)
    acc = accuracy_score(y_test, modelo.predict(X_test))
    print(f"[RESULTADO] Acurácia do Modelo {prefixo_liga.upper()}: {acc * 100:.2f}%")
    
    return modelo, historico_features, acc

# === FUNÇÕES DE INTERFACE ===

def encontrar_jogador_proximo(nome_input, nomes_base):
    nome, score = process.extractOne(nome_input, nomes_base)
    return nome if score >= 85 else None

def prever_partida_unica(modelo, historico_features):
    print("\n=== SIMULAÇÃO DE PARTIDA ===")
    j1 = input("Digite o nome do(a) Jogador(a) 1: ").strip()
    j2 = input("Digite o nome do(a) Jogador(a) 2: ").strip()

    nomes_base = historico_features.keys()
    nome1 = encontrar_jogador_proximo(j1, nomes_base)
    nome2 = encontrar_jogador_proximo(j2, nomes_base)

    if not nome1 or not nome2:
        print("[ERRO] Jogadores não encontrados ou sem histórico de estatísticas detalhadas.")
        return

    hist1 = historico_features.get(nome1, [])
    hist2 = historico_features.get(nome2, [])

    if not hist1 or not hist2:
         print(f"[AVISO] Histórico insuficiente para um dos jogadores. A previsão pode ser menos precisa.")
         return

    forma1 = pd.DataFrame(hist1).mean()
    forma2 = pd.DataFrame(hist2).mean()

    features_previsao = []
    feature_cols = [
        'serve_pts_won_pct', 'ace_pct', 'df_pct', 'first_serve_in_pct', 
        'first_serve_win_pct', 'second_serve_win_pct', 'bp_saved_pct', 'bp_converted_pct'
    ]
    for col in feature_cols:
        diff = forma1.get(col, 0) - forma2.get(col, 0)
        features_previsao.append(diff)
        
    diff = np.array([features_previsao])

    prob = modelo.predict_proba(diff)[0][1]
    vencedor = nome1 if prob > 0.5 else nome2

    print("\n=== RESULTADO DA PREVISÃO ===")
    print(f"Confronto: {nome1} vs {nome2}")
    print(f"Prob. {nome1}: {prob*100:.1f}% | Prob. {nome2}: {(1-prob)*100:.1f}%")
    print(f"=> Vencedor Previsto: {vencedor}")

# === MENU INTERATIVO ===

def menu_principal():
    modelos = {}

    while True:
        print("\n==== MENU PRINCIPAL ====")
        liga = input("Escolha a liga:\n1. ATP\n2. WTA\n>> ").strip()
        if liga not in ['1', '2']:
            print("[ERRO] Escolha inválida.")
            continue

        prefixo = 'atp' if liga == '1' else 'wta'
        if prefixo not in modelos:
            print("[INFO] Treinando modelo especialista. Isso pode levar um momento...")
            modelos[prefixo] = carregar_e_treinar(prefixo)

        if not modelos.get(prefixo) or not modelos[prefixo][0]:
            print(f"[ERRO] Falha ao treinar o modelo para {prefixo.upper()}. Verifique os arquivos na pasta 'database'.")
            continue
            
        modelo, hist, acc = modelos[prefixo]

        while True:
            acao = input(
                f"\n--- LIGA: {prefixo.upper()} (Acurácia: {acc * 100:.2f}%) ---\n"
                "1. Simular Partida\n2. Trocar Liga\n3. Sair\n>> "
            ).strip()
            if acao == '1':
                prever_partida_unica(modelo, hist)
            elif acao == '2':
                break
            elif acao == '3':
                print("\n[INFO] Encerrando o simulador.")
                return
            else:
                print("[ERRO] Opção inválida.")

# === EXECUÇÃO ===
if __name__ == '__main__':
    menu_principal()
