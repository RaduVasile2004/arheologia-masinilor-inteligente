import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from main import KantorovichProjectSolver  # Importăm clasa pentru a refolosi logica de încărcare


def generate_plots():
    # 1. Încărcăm datele folosind logica existentă
    solver = KantorovichProjectSolver()
    solver.load_files()
    solver.process_all()  # Rulăm din nou pentru a popula results_log

    # Transformăm log-ul în DataFrame pentru manipulare ușoară
    df = pd.DataFrame(solver.results_log)

    # Curățăm datele (eliminăm N/A sau valori lipsă pentru grafice)
    df = df[df['Status'] != 'N/A']
    df['Result'] = pd.to_numeric(df['Result'])
    df['Expected'] = pd.to_numeric(df['Expected'])
    df['Time'] = pd.to_numeric(df['Time'])

    # Setăm stilul vizual
    sns.set_theme(style="whitegrid")

    # --- GRAFIC 1: VALIDARE (CALCULAT vs ASTEPTAT) ---
    # Filtrăm doar câteva probleme reprezentative pentru a nu aglomera graficul
    subset = df[df['Category'].isin(['Problem A (Gen)', 'Problem B (Gen)'])].head(15)

    plt.figure(figsize=(12, 6))

    # Pregătim datele pentru format "long" (necesar pt Seaborn grouped bar)
    df_melted = subset.melt(id_vars=['ID'], value_vars=['Result', 'Expected'],
                            var_name='Tip', value_name='Valoare')

    sns.barplot(data=df_melted, x='ID', y='Valoare', hue='Tip', palette=['#2ecc71', '#95a5a6'])

    plt.title('Validare Rezultate: Algoritm Modern (Calculat) vs. Referință', fontsize=14)
    plt.ylabel('Total Output / Z')
    plt.xlabel('ID Problemă')
    plt.legend(title='')
    plt.tight_layout()
    plt.savefig('grafic_validare.png')
    print("✅ Generat: grafic_validare.png")

    # --- GRAFIC 2: TIMP DE EXECUȚIE ---
    plt.figure(figsize=(10, 5))

    # Sortăm după timp pentru a vedea trendul
    df_sorted = df.sort_values('Time')

    sns.lineplot(data=df_sorted, x='ID', y='Time', marker='o', color='#e74c3c', linewidth=2)

    plt.title('Performanța Algoritmului: Timp de Execuție per Problemă', fontsize=14)
    plt.ylabel('Timp (ms)')
    plt.xlabel('Probleme (ordonate după complexitate/timp)')
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.savefig('grafic_performanta.png')
    print("✅ Generat: grafic_performanta.png")

    # --- GRAFIC 3: HEATMAP KANTOROVICH ORIGINAL ---
    # Luăm datele manual pentru problema originală (pentru exemplu)
    # Rezultatul optim calculat:
    # M1 (Freze): 86.6% P1, 13.3% P2
    # M2 (Strung): 100% P1
    # M3 (Automat): 100% P2

    # Matricea h_ik (timp alocat)
    allocation_matrix = [
        [0.867, 0.133],  # Mașina 1
        [1.000, 0.000],  # Mașina 2
        [0.000, 1.000]  # Mașina 3
    ]

    plt.figure(figsize=(6, 5))
    sns.heatmap(allocation_matrix, annot=True, cmap="YlGnBu", fmt=".1%",
                xticklabels=['Piesa 1', 'Piesa 2'],
                yticklabels=['Freze', 'Strunguri', 'Automate'])

    plt.title('Alocarea Optimă a Timpului (Problem A)', fontsize=14)
    plt.tight_layout()
    plt.savefig('grafic_heatmap_kantorovich.png')
    print("✅ Generat: grafic_heatmap_kantorovich.png")


if __name__ == "__main__":
    generate_plots()