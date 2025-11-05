import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from pathlib import Path
import pickle

# --- Absolute Path Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "best_gat_realistic" / "data.pkl"

# --- Model & Data Configuration ---
NUM_ANTIBIOTICS = 30  # Based on amr_labels_realistic_distribution.csv

# --- Model Class (with GATConv Fixes) ---
class GAT_Light(nn.Module):
    def __init__(self, num_abs):
        super().__init__()
        # FIX: Use explicit keyword arguments for GATConv
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
        x = torch.mean(x, dim=0, keepdim=True)  # Global Mean Pool
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

# --- Helper Functions ---
def load_amr_model() -> GAT_Light:
    """Loads the GAT model with a custom unpickler to handle legacy files."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at path: {MODEL_PATH}")

    print(f"Loading model from {MODEL_PATH}...")
    model = GAT_Light(num_abs=NUM_ANTIBIOTICS)

    # Custom Unpickler to handle the 'persistent_load' instruction
    class CustomUnpickler(pickle.Unpickler):
        def persistent_load(self, saved_id):
            return None

    # A custom class that mimics the `pickle` module's interface
    class CustomPickleModule:
        __name__ = "custom_pickle"
        Unpickler = CustomUnpickler

        @staticmethod
        def load(f):
            # The load function that torch.load will call
            return CustomPickleModule.Unpickler(f).load()

    # Load the state dictionary using the custom pickle module class
    state_dict = torch.load(
        MODEL_PATH,
        map_location=torch.device('cpu'),
        pickle_module=CustomPickleModule  # Pass the class, not an instance
    )

    # Clean up the state dictionary keys (e.g., remove 'module.' prefix)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    # Load the state into the model
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print("Model loaded successfully inside inference.py.")
    return model
