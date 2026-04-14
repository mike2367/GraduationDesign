from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv, MessagePassing
from torch_geometric.utils import softmax


class AttentionRGCNConv(MessagePassing):
    """
    Relation-specific graph convolution with learnable relation-type attention.
    
    **Why custom implementation?**
    PyTorch Geometric provides:
    - ``GATConv``: node-pair attention (learns α per edge)
    - ``RGCNConv``: multi-relational conv without attention
    - **No built-in**: relation-type-level attention (one α per relation type)
    
    This layer learns a global attention coefficient α_r for each relation type r,
    allowing us to answer: "How important is STRING_association vs TF_regulates
    for SL prediction?" This is more interpretable than edge-level attention for
    knowledge graph analysis where we want transferable relation importance scores.
    
    Architecture:
    - Basis decomposition (like RGCNConv) for efficiency: W_r = Σ_b α_{r,b} B_b
    - Per-relation attention: α_r ∈ ℝ (one scalar per relation type, softmax-normalized)
    - Message aggregation: m_i = Σ_{j∈N(i)} α_{r(i,j)} · W_{r(i,j)} h_j
    
    Attention scores are cached in ``_last_attention_weights`` for extraction.
    """

    def __init__(self, in_channels: int, out_channels: int, num_relations: int, num_bases: Optional[int] = None):
        super().__init__(aggr="add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases or num_relations

        # Basis decomposition for efficiency
        self.basis = nn.Parameter(torch.Tensor(self.num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, self.num_bases))  # relation → basis attention
        self.root = nn.Linear(in_channels, out_channels, bias=False)
        
        # Attention: learnable per-relation importance (scalar)
        self.relation_attention = nn.Parameter(torch.ones(num_relations))
        
        self.reset_parameters()
        self._last_attention_weights = None  # store for extraction

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.basis)
        nn.init.xavier_uniform_(self.att)
        self.root.reset_parameters()
        nn.init.constant_(self.relation_attention, 1.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        # Compute relation-specific weight matrices
        weight = torch.einsum("rb,bio->rio", self.att, self.basis)  # [R, in, out]
        
        # Normalize attention across relations (softmax for interpretability)
        attn = F.softmax(self.relation_attention, dim=0)
        self._last_attention_weights = attn.detach().cpu().numpy()  # save for analysis
        
        out = self.propagate(edge_index, x=x, edge_type=edge_type, weight=weight, attn=attn)
        out = out + self.root(x)  # Add self-loop transformation
        return out

    def message(self, x_j: torch.Tensor, edge_type: torch.Tensor, weight: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        # x_j: [E, in], edge_type: [E], weight: [R, in,out], attn: [R]
        w = weight[edge_type]  # [E, in, out]
        a = attn[edge_type].unsqueeze(-1)  # [E, 1]
        msg = torch.bmm(x_j.unsqueeze(1), w).squeeze(1)  # [E, out]
        return msg * a  # scale by attention


class RGCN_SL_Predictor(nn.Module):
    """
    Multi-layer R-GCN encoder + deep MLP decoder for synthetic lethality
    prediction, with a **weight-extraction API** that produces edge-type /
    source / node-type importance dicts consumed by the heuristic
    graph-search pipeline (algo_config).

    Architecture
    ------------
    * Configurable depth (default 4) with residual connections at every layer.
    * LayerNorm for stable training on small heterogeneous graphs.
    * Decoder input: ``[z_i; z_j; z_i ⊙ z_j]`` capturing both shared and
      complementary features between gene pair embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_relations: int,
        num_layers: int = 4,
        num_bases: int | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.dropout = dropout
        self.use_attention = True  # flag to enable attention-based RGCN

        # ── encoder ──────────────────────────────────────────────────────
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_projs = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = out_channels if i == num_layers - 1 else hidden_channels
            if self.use_attention:
                self.convs.append(
                    AttentionRGCNConv(in_ch, out_ch, num_relations, num_bases=num_bases)
                )
            else:
                self.convs.append(
                    RGCNConv(in_ch, out_ch, num_relations, num_bases=num_bases)
                )
            self.norms.append(nn.LayerNorm(out_ch))
            self.res_projs.append(
                nn.Linear(in_ch, out_ch, bias=False)
                if in_ch != out_ch
                else nn.Identity()
            )

        # ── decoder ──────────────────────────────────────────────────────
        # [z_i; z_j; z_i ⊙ z_j] → 4-layer MLP
        dec_in = out_channels * 3
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.LayerNorm(hidden_channels // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels // 4, 1),
        )

    # ── forward ──────────────────────────────────────────────────────────

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        h = x
        for i, (conv, norm, res) in enumerate(
            zip(self.convs, self.norms, self.res_projs)
        ):
            h_in = h
            h = conv(h, edge_index, edge_type)
            h = norm(h)
            h = h + res(h_in)                               # residual
            if i < self.num_layers - 1:                      # no act on final layer
                h = F.gelu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def decode_pairs(
        self, z: torch.Tensor, gene_pairs: torch.Tensor
    ) -> torch.Tensor:
        zi, zj = z[gene_pairs[0]], z[gene_pairs[1]]
        # Order-invariant: SL is symmetric, so (a,b) ≡ (b,a)
        pair_emb = torch.cat([zi + zj, zi * zj, torch.abs(zi - zj)], dim=-1)
        return self.decoder(pair_emb).squeeze(-1)

    def forward(
        self, data: Data, gene_pairs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(data.x, data.edge_index, data.edge_type)
        return self.decode_pairs(z, gene_pairs), z

    # ═════════════════════════════════════════════════════════════════════
    # Weight-extraction API — produces dicts that algo_config consumes
    # ═════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _relation_norms(self) -> Dict[int, float]:
        """Mean Frobenius norm of per-relation weight matrices across layers."""
        agg: Dict[int, List[float]] = {}
        for conv in self.convs:
            if isinstance(conv, AttentionRGCNConv):
                # Use learned attention scores directly
                if conv._last_attention_weights is not None:
                    for r in range(self.num_relations):
                        agg.setdefault(r, []).append(float(conv._last_attention_weights[r]))
                else:
                    # Fallback: use raw attention parameters
                    attn = F.softmax(conv.relation_attention, dim=0).cpu().numpy()
                    for r in range(self.num_relations):
                        agg.setdefault(r, []).append(float(attn[r]))
            else:
                # Original norm-based extraction for backward compat
                W = conv.weight  # [B, in, out] or [R, in, out]
                comp = getattr(conv, "comp", None)  # [R, B] when basis decomp
                R = self.num_relations if comp is not None else W.shape[0]
                for r in range(R):
                    if comp is not None:
                        W_r = torch.einsum("b, bio -> io", comp[r], W)
                    else:
                        W_r = W[r]
                    agg.setdefault(r, []).append(W_r.norm().item())
        return {r: float(np.mean(v)) for r, v in agg.items()}
    
    @torch.no_grad()
    def get_attention_scores(self) -> Dict[int, List[float]]:
        """Extract raw attention scores from all attention-based layers."""
        self.eval()
        attn_by_relation: Dict[int, List[float]] = {}
        for conv in self.convs:
            if isinstance(conv, AttentionRGCNConv):
                attn = F.softmax(conv.relation_attention, dim=0).cpu().numpy()
                for r in range(self.num_relations):
                    attn_by_relation.setdefault(r, []).append(float(attn[r]))
        return attn_by_relation

    def extract_relation_weights(
        self, idx_to_rel: Dict[int, str]
    ) -> Dict[str, float]:
        """
        Learned relation-type importance → ``EDGE_RELATION_WEIGHT``.

        Aggregates conv weight norms by relation type (splits the compound
        ``type|||source`` keys).  Scaled to ≈[0, 2.0].
        """
        norms = self._relation_norms()
        by_type: Dict[str, List[float]] = {}
        for r_idx, norm_val in norms.items():
            compound = idx_to_rel.get(r_idx, f"unknown_{r_idx}")
            rel_type = compound.split("|||")[0]
            by_type.setdefault(rel_type, []).append(norm_val)
        avg = {k: float(np.mean(v)) for k, v in by_type.items()}
        # Return raw averages; calibration happens in train.py
        return {k: round(v, 6) for k, v in avg.items()}

    def extract_source_weights(
        self, idx_to_rel: Dict[int, str]
    ) -> Dict[str, float]:
        """
        Learned source reliability → ``EDGE_SOURCE_WEIGHT``.  Scaled to [0, 1].
        """
        norms = self._relation_norms()
        by_src: Dict[str, List[float]] = {}
        for r_idx, norm_val in norms.items():
            compound = idx_to_rel.get(r_idx, f"unknown_{r_idx}")
            parts = compound.split("|||")
            source = parts[1] if len(parts) > 1 else "unknown"
            by_src.setdefault(source, []).append(norm_val)
        avg = {k: float(np.mean(v)) for k, v in by_src.items()}
        # Return raw averages; calibration happens in train.py
        return {k: round(v, 6) for k, v in avg.items()}

    def extract_edge_type_priority(
        self, idx_to_rel: Dict[int, str]
    ) -> Dict[str, int]:
        """
        ``EDGE_TYPE_PRIORITY`` derived from learned relation weights
        (higher weight → lower priority number = more preferred).
        """
        rel_w = self.extract_relation_weights(idx_to_rel)
        return {
            t: rank
            for rank, (t, _) in enumerate(
                sorted(rel_w.items(), key=lambda x: -x[1])
            )
        }

    @torch.no_grad()
    def extract_node_type_ranks(
        self,
        data: Data,
        node_to_idx: Dict[str, int],
        nodes_df,
    ) -> Dict[str, int]:
        """
        ``NODE_TYPE_RANK``: higher avg ‖z‖ → lower rank number (more important).
        """
        self.eval()
        z = self.encode(data.x, data.edge_index, data.edge_type)
        norms = z.norm(dim=1).cpu().numpy()

        type_norms: Dict[str, List[float]] = {}
        for _, row in nodes_df.iterrows():
            idx = node_to_idx.get(row["node_id"])
            if idx is not None and idx < len(norms):
                ntype = str(row["type"]).strip().lower()
                type_norms.setdefault(ntype, []).append(float(norms[idx]))

        avg = {k: float(np.mean(v)) for k, v in type_norms.items()}
        sorted_types = sorted(avg.items(), key=lambda x: -x[1])
        return {t: rank + 1 for rank, (t, _) in enumerate(sorted_types)}

    def extract_all_weights(
        self,
        data: Data,
        node_to_idx: Dict[str, int],
        idx_to_rel: Dict[int, str],
        nodes_df,
    ) -> Dict[str, dict]:
        """Extract all learned weights as a dict ready for JSON serialisation."""
        rel_w = self.extract_relation_weights(idx_to_rel)
        return {
            "EDGE_RELATION_WEIGHT": rel_w,
            "EDGE_TYPE_WEIGHT": rel_w,  # same semantics for path construction
            "EDGE_SOURCE_WEIGHT": self.extract_source_weights(idx_to_rel),
            "EDGE_TYPE_PRIORITY": self.extract_edge_type_priority(idx_to_rel),
            "NODE_TYPE_RANK": self.extract_node_type_ranks(
                data, node_to_idx, nodes_df
            ),
        }
