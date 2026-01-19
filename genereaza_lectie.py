import json

# Codul curat al clasei Solver (pentru a fi inserat in notebook)
solver_code = """import pulp
import numpy as np
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class KantorovichProjectSolver:
    def __init__(self):
        self.results_log = []

    def solve_allocation(self, data, problem_type='A'):
        # 1. Extragere Date
        prob_id = data.get('id', data.get('name', 'Unknown'))
        raw_prod = data.get('productivity') or data.get('productivity_matrix')
        prod = np.array(raw_prod)
        rows, cols = prod.shape

        # Gestionare Capacitate Masini (ex: 3 freze = capacitate 3.0)
        machine_capacities = [1.0] * rows
        if 'machines' in data and isinstance(data['machines'], list):
            for idx, m in enumerate(data['machines']):
                if isinstance(m, dict) and 'count' in m and idx < rows:
                    machine_capacities[idx] = float(m['count'])

        # 2. Modelare PuLP
        model = pulp.LpProblem(f"Alloc_{prob_id}", pulp.LpMaximize)
        Z = pulp.LpVariable("Z", lowBound=0)
        h = pulp.LpVariable.dicts("h", ((i, j) for i in range(rows) for j in range(cols)), lowBound=0)

        model += Z

        # Constrangere: Capacitate Masini
        for i in range(rows):
            model += pulp.lpSum([h[(i, j)] for j in range(cols)]) == machine_capacities[i]

        # Constrangere: Balanta Productie
        for j in range(cols):
            model += pulp.lpSum([prod[i][j] * h[(i, j)] for i in range(rows)]) == Z

        # Constrangere: Resurse (Tip B)
        if problem_type == 'B':
            cons = data.get('resource_consumption') or data.get('energy_consumption')
            limit = data.get('max_resource') or data.get('max_energy')
            if cons and limit:
                try:
                    # Sum(consum_specific * timp_alocat) <= Limita
                    model += pulp.lpSum([cons[i][j] * h[(i, j)] for i in range(rows) for j in range(cols)]) <= limit
                except:
                    pass

        # 3. Rezolvare
        start = time.time()
        model.solve(pulp.PULP_CBC_CMD(msg=False))

        # Pregatire date pentru vizualizare
        allocation_matrix = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                allocation_matrix[i][j] = pulp.value(h[(i,j)])

        return {
            "id": prob_id,
            "val": pulp.value(Z),
            "time": (time.time() - start) * 1000,
            "allocation": allocation_matrix,
            "machines": data.get('machine_names', [f"M{i+1}" for i in range(rows)]),
            "parts": data.get('part_names', [f"P{j+1}" for j in range(cols)])
        }

    def plot_allocation(self, result):
        plt.figure(figsize=(10, 6))
        sns.heatmap(result['allocation'], annot=True, cmap="YlGnBu", fmt=".2f",
                    xticklabels=result['parts'], yticklabels=result['machines'])
        plt.title(f"Alocarea Optima: {result['id']} (Z = {result['val']:.2f})", fontsize=14)
        plt.xlabel("Piese (Produse)")
        plt.ylabel("Masini (Resurse)")
        plt.show()
"""

# Structura Notebook-ului
notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Arheologia Mașinilor Inteligente: De la Kantorovich (1939) la Python\n",
                "\n",
                "**Material Didactic** | **Durata:** 45 min  \n",
                "**Subiect:** Programare Liniară, Optimizare de Resurse, Istoria Algoritmilor\n",
                "\n",
                "---\n",
                "\n",
                "## 1. Introducere și Context Istoric\n",
                "\n",
                "În 1939, matematicianul sovietic **Leonid Kantorovich** a publicat lucrarea *\"Metode matematice de organizare și planificare a producției\"*. Într-o fabrică de placaj din Leningrad, el a observat că mașinile nu erau folosite eficient. Deși inginerii încercau să maximizeze producția, le lipsea o metodă matematică riguroasă pentru a aloca resursele.\n",
                "\n",
                "Kantorovich a inventat ceea ce numim astăzi **Programare Liniară (Linear Programming)**, o metodă pentru a obține cel mai bun rezultat (profit maxim sau cost minim) într-un model matematic reprezentat prin relații liniare. Pentru această descoperire, a primit Premiul Nobel pentru Economie în 1975.\n",
                "\n",
                "### Ce vom face astăzi?\n",
                "1. Vom înțelege matematica din spatele **Problemei A** (Alocarea Mașinilor).\n",
                "2. Vom folosi un solver modern (Python + PuLP) pentru a rezolva problema.\n",
                "3. Vom compara rezultatele moderne cu cele calculate manual în 1939.\n",
                "\n",
                ""
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Modelul Matematic (Teorie)\n",
                "\n",
                "Să presupunem că avem o fabrică cu mai multe tipuri de mașini și trebuie să producem seturi complete de piese (ex: un șurub și o piuliță formează un set).\n",
                "\n",
                "### Variabilele\n",
                "Notăm cu $h_{ik}$ fracțiunea de timp pe care mașina $i$ o petrece lucrând la piesa $k$.\n",
                "* Dacă $h_{11} = 0.5$, înseamnă că Mașina 1 lucrează 50% din timp la Piesa 1.\n",
                "\n",
                "### Funcția Obiectiv\n",
                "Vrem să maximizăm numărul total de seturi complete, notat cu $Z$.\n",
                "$$ \\max Z $$\n",
                "\n",
                "### Constrângerile (Regulile Jocului)\n",
                "\n",
                "1. **Limita de Timp:** Fiecare mașină are o capacitate limitată (100% din timp). Dacă avem 3 mașini identice de tip $i$, capacitatea este 3.0.\n",
                "   $$ \\sum_{k} h_{ik} = \\text{Număr Mașini}_i $$\n",
                "\n",
                "2. **Balanța Producției:** Nu ne ajută să producem 1000 de șuruburi dacă avem doar 10 piulițe. Numărul de piese de tip $k$ produse trebuie să fie egal cu numărul total de seturi $Z$.\n",
                "   $$ \\sum_{i} (\\text{Productivitate}_{ik} \\times h_{ik}) = Z $$\n",
                "\n",
                "3. **Non-negativitate:** Nu putem aloca timp negativ.\n",
                "   $$ h_{ik} \\ge 0 $$"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Importam bibliotecile necesare\n",
                "# PuLP este biblioteca standard in Python pentru Programare Liniara\n",
                "try:\n",
                "    import pulp\n",
                "    import numpy as np\n",
                "    import pandas as pd\n",
                "    import seaborn as sns\n",
                "    import matplotlib.pyplot as plt\n",
                "    print(\"Biblioteci incarcate cu succes!\")\n",
                "except ImportError:\n",
                "    print(\"Instalati bibliotecile necesare: pip install pulp numpy pandas seaborn matplotlib\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Motorul de Calcul (Solver)\n",
                "\n",
                "Aceasta este clasa care transformă ecuațiile matematice de mai sus în cod Python. Folosim algoritmul **Simplex** (prin intermediul `PuLP`), care navighează prin colțurile poligonului de soluții posibile pentru a găsi optimul."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"{solver_code}"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Studiu de Caz: Problema Originală (Kantorovich, 1939)\n",
                "\n",
                "Să recreăm exemplul din carte. Avem:\n",
                "* **3 Freze (Milling Machines):** Rapide la Piesa 2, lente la Piesa 1.\n",
                "* **3 Strunguri (Lathe Machines):** Echilibrate.\n",
                "* **1 Automat:** Foarte rapid la Piesa 2.\n",
                "\n",
                "Productivitatea este dată în piese/oră."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Definim datele problemei istorice\n",
                "problem_kantorovich = {\n",
                "    \"id\": \"Kantorovich_Problem_A\",\n",
                "    \"name\": \"Alocare Masini (1939)\",\n",
                "    \"machines\": [\n",
                "      {\"name\": \"Freze\", \"count\": 3},\n",
                "      {\"name\": \"Strunguri\", \"count\": 3},\n",
                "      {\"name\": \"Automate\", \"count\": 1}\n",
                "    ],\n",
                "    \"machine_names\": [\"Freze (3 buc)\", \"Strunguri (3 buc)\", \"Automat (1 buc)\"],\n",
                "    \"part_names\": [\"Piesa 1\", \"Piesa 2\"],\n",
                "    # Productivitate [Masina][Piesa]\n",
                "    \"productivity\": [\n",
                "      [10, 20],  # Freze: fac 10 piese1 sau 20 piese2\n",
                "      [20, 30],  # Strunguri\n",
                "      [30, 80]   # Automat: foarte eficient la piesa 2\n",
                "    ]\n",
                "}\n",
                "\n",
                "# Initializam solverul\n",
                "solver = KantorovichProjectSolver()\n",
                "\n",
                "# Rezolvam\n",
                "rezultat = solver.solve_allocation(problem_kantorovich, problem_type='A')\n",
                "\n",
                "print(f\"\\n--- REZULTATE ---\")\n",
                "print(f\"Productia Maxima (Z): {rezultat['val']:.4f} seturi complete\")\n",
                "print(f\"Timp de calcul: {rezultat['time']:.2f} milisecunde\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Analiza Vizuală a Soluției\n",
                "\n",
                "Graficul de mai jos (Heatmap) ne arată strategia optimă:\n",
                "* Culorile închise indică o concentrare mare de resurse.\n",
                "* Observați cum solverul \"specializează\" mașinile. De exemplu, Automatul ar trebui să lucreze aproape exclusiv la Piesa 2, unde are avantaj comparativ maxim."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "solver.plot_allocation(rezultat)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Discuție: 86.00 vs 86.67\n",
                "\n",
                "În cartea sa, Kantorovich ajunge la soluția de **86** de seturi. Algoritmul nostru modern arată **86.6667**.\n",
                "\n",
                "**De ce apare diferența?**\n",
                "Algoritmul Simplex lucrează cu numere reale continue (Programare Liniară). Matematic, optimul este $260/3 \\approx 86.67$. Totuși, în realitate, nu poți livra 0.67 dintr-un set. Kantorovich, calculând manual sau folosind metode numerice timpurii, a rotunjit la cel mai apropiat număr întreg fezabil.\n",
                "\n",
                "Acest lucru demonstrează acuratețea superioară a uneltelor digitale moderne."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Avansat: Problema cu Constrângeri Multiple (Tip B)\n",
                "\n",
                "În realitate, nu doar timpul mașinii este limitat. Putem avea limite de energie electrică, materie primă sau forță de muncă.\n",
                "\n",
                "Să modificăm problema: Adăugăm un consum de energie. Mașinile rapide consumă mai mult curent.\n",
                "* Avem un buget de **150 kWh**.\n",
                "* Frezele consumă 5 kWh/oră, Automatul consumă 15 kWh/oră."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "problem_with_energy = {\n",
                "    \"id\": \"Problem_B_Energy_Constraint\",\n",
                "    \"machines\": [\n",
                "      {\"name\": \"Freze\", \"count\": 3},\n",
                "      {\"name\": \"Strunguri\", \"count\": 3},\n",
                "      {\"name\": \"Automate\", \"count\": 1}\n",
                "    ],\n",
                "    \"productivity\": [\n",
                "      [10, 20],\n",
                "      [20, 30],\n",
                "      [30, 80]\n",
                "    ],\n",
                "    # Consum energie [kWh per unitate de timp alocata]\n",
                "    \"resource_consumption\": [\n",
                "        [5, 5],    # Frezele consuma putin\n",
                "        [8, 8],    # Strungurile mediu\n",
                "        [15, 20]   # Automatul consuma mult, mai ales la piesa 2\n",
                "    ],\n",
                "    \"max_resource\": 150, # Limita totala de energie\n",
                "    \"machine_names\": [\"Freze\", \"Strunguri\", \"Automat\"],\n",
                "    \"part_names\": [\"Piesa 1\", \"Piesa 2\"]\n",
                "}\n",
                "\n",
                "rezultat_B = solver.solve_allocation(problem_with_energy, problem_type='B')\n",
                "\n",
                "print(f\"Productia cu limita de energie: {rezultat_B['val']:.4f} seturi\")\n",
                "solver.plot_allocation(rezultat_B)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Concluzii\n",
                "\n",
                "1. **Eficiență:** Programarea liniară permite găsirea soluției optime în milisecunde, o sarcină care ar dura ore întregi manual.\n",
                "2. **Specializare:** Matematica ne arată că eficiența maximă se obține prin specializarea mașinilor (avantaj comparativ), nu prin a pune fiecare mașină să facă \"câte puțin din toate\".\n",
                "3. **Scalabilitate:** Același cod poate rezolva o problemă cu 1000 de mașini și 500 de tipuri de piese fără modificări majore."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Scriere fisier
with open('Lectia_Kantorovich_Advanced.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=2)

print("Succes! Fisierul 'Lectia_Kantorovich_Advanced.ipynb' a fost generat.")