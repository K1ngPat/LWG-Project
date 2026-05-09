import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder


@register_node_encoder('GraphStatsSE')
class GraphStatsSENodeEncoder(torch.nn.Module):
    """Graph-level structural statistics encoder.

    The preprocessing stage computes graph-wide statistics once per graph and
    broadcasts them to all nodes. This encoder then projects the raw statistics
    into the requested embedding dimension and concatenates them to the node
    features.
    """

    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        dim_in = cfg.share.dim_in

        pecfg = cfg.posenc_GraphStatsSE
        dim_pe = pecfg.dim_pe
        model_type = pecfg.model.lower()
        if model_type not in ['linear', 'deepset', 'mlp']:
            raise ValueError(f"Unexpected PE model {pecfg.model}")
        self.model_type = model_type
        n_layers = pecfg.layers
        norm_type = pecfg.raw_norm_type.lower()
        self.pass_as_var = pecfg.pass_as_var

        if dim_emb - dim_pe < 0:
            raise ValueError(f"GraphStatsSE size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        # Determine how many raw stats are enabled in the config
        enabled = getattr(cfg.posenc_GraphStatsSE, 'enabled_stats', None)
        if enabled is None or len(enabled) == 0:
            self.num_stats = 4
        else:
            self.num_stats = len(enabled)
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(self.num_stats)
        else:
            self.raw_norm = None

        activation = nn.ReLU
        if self.model_type == 'linear':
            self.pe_encoder = nn.Linear(self.num_stats, dim_pe)
        else:
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(self.num_stats, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(self.num_stats, 2 * dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)

    def forward(self, batch):
        pestat_var = 'pestat_GraphStatsSE'
        if not hasattr(batch, pestat_var):
            raise ValueError(f"Precomputed '{pestat_var}' variable is required "
                             f"for {self.__class__.__name__}; set config "
                             "'posenc_GraphStatsSE.enable' to True")

        pos_enc = getattr(batch, pestat_var)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)

        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x

        batch.x = torch.cat((h, pos_enc), 1)
        if self.pass_as_var:
            batch.pe_GraphStatsSE = pos_enc
        return batch