import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# --- CONFIGURATION AND FILE PATHS ---
# Final file name determined by successful data loading
FEATURE_FILE = "ncbi_features.csv" 
LABEL_FILE = "amr_labels_realistic_distribution.csv"

# CRITICAL FIX: Path points to the file *inside* the folder
MODEL_PATH = "best_gat_realistic/data.pkl" 

# --- MODEL CLASS ---
class GAT_Light(nn.Module):
    """Graph Attention Network model for AMR prediction."""
    def __init__(self, num_abs):
        super().__init__()
        # FIX: Explicitly name in_channels and out_channels (solves initial TypeError)
        self.gat1 = GATConv(in_channels=1, out_channels=16, heads=4, dropout=0.5)
        self.gat2 = GATConv(in_channels=16*4, out_channels=8, heads=2, dropout=0.5)
        
        self.fc1 = nn.Linear(8*2, 32) 
        self.fc2 = nn.Linear(32, num_abs)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.dropout(x)
        x = self.relu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.gat2(x, edge_index))

        # Global mean pool
        x = torch.mean(x, dim=0, keepdim=True)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

# --- DATA AND MODEL LOADING FUNCTIONS ---

def load_data():
    """Loads feature and label dataframes and cleans the index."""
    print("Loading data...")
    if not all(os.path.exists(f) for f in [FEATURE_FILE, LABEL_FILE]):
        raise FileNotFoundError("One or more required data files (features or labels) are missing.")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at expected path: {MODEL_PATH}. "
                                 "Please ensure 'best_gat_realistic.pth' is a folder and 'data.pkl' is inside it.")

    # Load feature data
    features_df = pd.read_csv(FEATURE_FILE).set_index('genome_id')
    
    # CRITICAL FIX: Clean the index (genome_id) to remove hidden whitespace (solves 404 error)
    features_df.index = features_df.index.astype(str).str.strip() 
    
    # Load label data
    labels_df = pd.read_csv(LABEL_FILE).set_index('genome_id')

    antibiotics = list(labels_df.columns)
    
    print(f"Data loaded. Total genomes: {len(features_df)}")
    print(f"Antibiotics detected: {len(antibiotics)}")
    return features_df, antibiotics

def load_model(num_abs: int) -> GAT_Light:
    """Initializes and loads the trained model weights."""
    print(f"Initializing model and loading weights from {MODEL_PATH}...")
    model = GAT_Light(num_abs=num_abs)
    
    # CRITICAL FIX: Set weights_only=False to load complex saved model
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    
    # Adapt the state dict keys
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    # strict=False allows for minor differences in model structure
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print("Model loaded successfully.")
    return model

def create_graph_data(genome_id: str, features_df: pd.DataFrame) -> Data:
    """Creates a PyG Data object for a single genome prediction."""
    # Ensure the queried ID is cleaned before lookup
    cleaned_genome_id = genome_id.strip()

    if cleaned_genome_id not in features_df.index:
        raise HTTPException(status_code=404, detail=f"Genome ID '{genome_id}' not found in feature data. "
                                                     "Check if the ID is valid or if the data file is correct.")

    features = features_df.loc[cleaned_genome_id].values.astype(np.float32)
    x = torch.tensor(features).unsqueeze(1) 

    num_nodes = x.size(0)
    
    # Edge Index: Self-loops for all nodes
    edge_index = torch.arange(num_nodes, dtype=torch.long).repeat(2, 1)

    data = Data(x=x, edge_index=edge_index)
    return data

# --- FASTAPI SETUP ---
app = FastAPI(title="AMR Prediction Service", version="1.0")

# Global variables for data and model
try:
    features_df, antibiotics = load_data()
    model = load_model(num_abs=len(antibiotics))
except Exception as e:
    print(f"Startup Failure: {e}")
    features_df, antibiotics, model = None, None, None
    
# --- API SCHEMA ---
class PredictionRequest(BaseModel):
    """Input structure for the prediction endpoint."""
    genome_id: str = "GCA_002853715.1_ASM285371v1" 

class PredictionResult(BaseModel):
    """Output structure for a single antibiotic prediction."""
    antibiotic: str
    confidence: float
    prediction: str

class PredictionResponse(BaseModel):
    """Overall output structure for the API endpoint."""
    genome_id: str
    predictions: List[PredictionResult]
    summary: Dict[str, Any]

# --- API ENDPOINT ---
@app.post("/predict", response_model=PredictionResponse)
def predict_amr(request: PredictionRequest):
    """
    Predicts Antimicrobial Resistance (AMR) for a given genome ID.
    """
    if not model or features_df is None:
        raise HTTPException(status_code=503, detail="Service not ready. Model or data failed to load during startup.")

    genome_id = request.genome_id

    try:
        # 1. Create the PyG Data object (includes ID cleaning)
        data = create_graph_data(genome_id, features_df)

        # 2. Run prediction
        with torch.no_grad():
            output = model(data)
            
        # CRITICAL FIX: Flatten the output tensor before converting to numpy 
        probabilities = output.flatten().numpy()

        # 3. Format results
        results = []
        resistant_count = 0
        susceptible_count = 0
        RESISTANCE_THRESHOLD = 0.5
        
        for ab, prob in zip(antibiotics, probabilities):
            # CRITICAL FIX: Use item() to extract the scalar float from the NumPy array (solves TypeError)
            # We use float(prob.item()) to ensure it's a standard Python float for Pydantic serialization
            confidence = float(prob.item())
            is_resistant = confidence > RESISTANCE_THRESHOLD
            
            results.append(PredictionResult(
                antibiotic=ab,
                confidence=confidence,
                prediction="RESISTANT" if is_resistant else "SUSCEPTIBLE"
            ))
            if is_resistant:
                resistant_count += 1
            else:
                susceptible_count += 1
        
        summary = {
            "total_antibiotics": len(antibiotics),
            "resistant": resistant_count,
            "susceptible": susceptible_count,
            "resistance_threshold": RESISTANCE_THRESHOLD
        }

        return PredictionResponse(genome_id=genome_id, predictions=results, summary=summary)

    except HTTPException as e:
        raise e
    except Exception as e:
        # Catch and report any remaining execution errors (e.g., shape mismatches)
        print(f"An unexpected error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed due to an internal server error: {str(e)}")

# Add a simple root endpoint for health check
@app.get("/")
def read_root():
    return {"status": "ok", "message": "AMR Prediction API is running."}
