# -*- coding: utf-8 -*-
"""
This is the main application file, based on the `inference.py` logic.
It accepts a .fasta file upload, calculates k-mers, runs the GAT
model, and returns predictions with a graph visualization.

This version includes fixes for:
1.  Absolute paths to resolve 'FileNotFoundError'.
2.  GATConv explicit arguments.
3.  torch.load 'weights_only=False' and 'strict=False'.
4.  numpy .item() and .flatten() for 'TypeError'.
5.  Removal of the 'scaler.pkl' dependency for testing.
6.  CRITICAL: str(fasta_path) for BioPython 'TypeError'.

***********************************************************************
** WARNING - NO SCALER **
The 'StandardScaler' (scaler.pkl) dependency is removed.
THE PREDICTIONS FROM THIS FILE WILL BE SCIENTIFICALLY INCORRECT.
***********************************************************************
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import pandas as pd
import numpy as np
import torch
from Bio import SeqIO
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import List
import itertools # Added for explicit import
from inference import load_amr_model, GAT_Light

# --- Absolute Path Configuration ---
# CRITICAL FIX: Use the script's own directory to build absolute paths
# This makes the script runnable from any location.
SCRIPT_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = SCRIPT_DIR / "uploads"
RESULT_DIR = SCRIPT_DIR / "results"

# Create directories for uploads and results
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# --- Model & Data Configuration ---
NUM_KMERS = 4096  # (4^6) - Assuming 6-mers

# List of antibiotics (from amr_labels_realistic_distribution.csv)
ANTIBIOTICS_LIST = [
    'amikacin', 'gentamicin', 'streptomycin', 'tobramycin', 'kanamycin', 
    'ampicillin', 'piperacillin', 'amoxicillin', 'cefazolin', 'cephalexin', 
    'ceftriaxone', 'cefotaxime', 'cefuroxime', 'cefepime', 'imipenem', 
    'meropenem', 'ertapenem', 'doripenem', 'carbapenem', 'ciprofloxacin', 
    'levofloxacin', 'moxifloxacin', 'erythromycin', 'azithromycin', 
    'tetracycline', 'tigecycline', 'vancomycin', 'chloramphenicol', 
    'fosfomycin', 'colistin'
]

# --- Helper Functions ---
def get_kmer_counts(fasta_path: Path, k: int = 6) -> np.ndarray:
    """Calculates k-mer counts for a given FASTA file."""
    print(f"Calculating {k}-mer counts for {fasta_path}...")
    kmer_dict = {}
    
    # Generate all possible k-mers (4^k)
    bases = ['A', 'T', 'C', 'G']
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    for kmer in all_kmers:
        kmer_dict[kmer] = 0

    # Count k-mers in the fasta file
    try:
        # CRITICAL FIX: Convert Path object to string for SeqIO.parse
        for record in SeqIO.parse(str(fasta_path), "fasta"):
            seq = str(record.seq).upper()
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                if kmer in kmer_dict:
                    kmer_dict[kmer] += 1
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing FASTA file: {e}")

    # Ensure the order is correct
    counts = np.array([kmer_dict[kmer] for kmer in all_kmers])
    
    if len(counts) != NUM_KMERS:
        raise HTTPException(
            status_code=500, 
            detail=f"K-mer count ({len(counts)}) does not match model expectation ({NUM_KMERS}). Check k-mer size."
        )
    print("K-mer counts calculated.")
    return counts

def create_graph_data(unscaled_features: np.ndarray) -> Data:
    """Creates a PyG Data object from feature counts."""
    # We are now feeding UN-SCALED features directly to the model
    x = torch.tensor(unscaled_features, dtype=torch.float32).unsqueeze(1)
    num_nodes = x.size(0)
    # Create self-loops for each k-mer node
    edge_index = torch.arange(num_nodes, dtype=torch.long).repeat(2, 1)
    return Data(x=x, edge_index=edge_index)

# --- FastAPI Application ---
app = FastAPI(title="AMR End-to-End Prediction Service (NO SCALER - TEST ONLY)")

# Load model on startup
model = load_amr_model()

@app.post("/predict_fasta")
def predict_from_fasta(file: UploadFile = File(...)):
    """
    Upload a .fasta genome file, predict AMR, and get a result graph.
    WARNING: Predictions are meaningless as the scaler is removed.
    """
    if not model:
        raise HTTPException(
            status_code=503, 
            detail="Service not ready. Model failed to load on startup. Check server logs for 'STARTUP FAILURE'."
        )

    # Save uploaded file
    upload_path = UPLOAD_DIR / file.filename
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    try:
        # 1. Calculate k-mer counts
        kmer_counts = get_kmer_counts(upload_path)

        # 2. Prepare features (NO SCALER)
        kmer_counts_reshaped = kmer_counts.reshape(1, -1)
        features_for_graph = kmer_counts_reshaped.flatten()

        # 3. Create graph data
        data = create_graph_data(features_for_graph)

        # 4. Run prediction
        with torch.no_grad():
            # FIX: Use flatten() to ensure 1D array
            pred_probs = model(data).flatten().numpy()

        # 5. Format results
        resistant_count = 0
        susceptible_count = 0
        predictions = []
        for ab, prob in zip(ANTIBIOTICS_LIST, pred_probs):
            # FIX: Use item() to convert numpy float to Python float
            confidence = float(prob.item())
            is_resistant = confidence > 0.5
            
            pred_str = "RESISTANT" if is_resistant else "SUSCEPTIBLE"
            predictions.append({
                "antibiotic": ab,
                "confidence": confidence,
                "prediction": pred_str
            })
            if is_resistant:
                resistant_count += 1
            else:
                susceptible_count += 1

        # --- 6. Create and save graph visualization ---
        G = nx.Graph()
        G.add_node("Genome", size=4000, color='#3366CC')
        colors = []
        for p in predictions:
            G.add_node(p['antibiotic'], size=1000)
            G.add_edge("Genome", p['antibiotic'])
            colors.append('#FF3333' if p['prediction'] == 'RESISTANT' else '#33DD33')

        fig, ax = plt.subplots(figsize=(18, 18))
        pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
        
        nx.draw_networkx_nodes(G, pos, nodelist=["Genome"], node_size=4000, node_color='#3366CC', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=ANTIBIOTICS_LIST, node_size=1000, node_color=colors, ax=ax)
        nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.6, ax=ax)
        
        labels = {n: n for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold', font_color='white', ax=ax)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF3333', edgecolor='black', label=f'Resistant - {resistant_count}'),
            Patch(facecolor='#33DD33', edgecolor='black', label=f'Susceptible - {susceptible_count}')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=14)
        ax.set_title(f'AMR Prediction for {file.filename}', fontsize=16, weight='bold')
        ax.axis('off')

        plt.tight_layout()
        # Use absolute path for saving
        graph_path = RESULT_DIR / f"{Path(file.filename).stem}_graph.png"
        plt.savefig(graph_path, dpi=200, bbox_inches='tight')
        plt.close()

        # Use absolute path for saving
        csv_path = RESULT_DIR / f"{Path(file.filename).stem}_predictions.csv"
        pd.DataFrame(predictions).to_csv(csv_path, index=False)

        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "summary": {
                "resistant": resistant_count,
                "susceptible": susceptible_count
            },
            "predictions": predictions,
            # Return string representations of the paths
            "result_csv_file": str(csv_path),
            "result_graph_file": str(graph_path)
        })

    except HTTPException as e:
        raise e  # Re-raise HTTP exceptions
    except Exception as e:
        print(f"Error during prediction for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "AMR Prediction API (Inference Mode) is running. "
                     "WARNING: Scaler is disabled. Predictions are for test only."}
