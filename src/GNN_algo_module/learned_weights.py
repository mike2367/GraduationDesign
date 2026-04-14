"""GNN-learned weights for algorithm module (auto-generated).

These weights are learned from the R-GCN SL predictor trained on the full KG.
They replace the hand-tuned defaults in gnn_config.py.
"""

from typing import Dict

# Learned relation importance (normalized L2 norms across all R-GCN layers)
EDGE_RELATION_WEIGHT: Dict[str, float] = {
    "DepMap_codependency": 0.5033,
    "OmniPath_interaction": 0.5019,
    "SL_pair": 1.0000,
    "STRING_association": 0.5037,
    "TF_regulates": 0.5105,
    "driver_in": 0.5123,
    "encodes": 0.4994,
    "in_pathway": 0.5019,
    "targets": 0.4991,
}

# Source reliability (stable heuristics, not learned)
EDGE_SOURCE_WEIGHT: Dict[str, float] = {
    "Reactome": 1.00,
    "curated": 0.98,
    "DoRothEA/OmniPath": 0.97,
    "DoRothEA": 0.97,
    "CollecTRI": 0.97,
    "UniProt/OmniPath": 0.96,
    "OmniPath": 0.96,
    "seed": 0.90,
    "IntOGen": 0.82,
    "cBioPortal": 0.78,
    "DepMap": 0.75,
    "OpenTargets": 0.70,
    "STRING": 0.55,
    "unknown": 0.30,
}

# Node type importance ranks (lower = more important)
NODE_TYPE_RANK: Dict[str, int] = {
    "cohort": 5,
    "drug": 4,
    "gene": 3,
    "other": 6,
    "pathway": 1,
    "protein": 2,
    "unknown": 6,
}
