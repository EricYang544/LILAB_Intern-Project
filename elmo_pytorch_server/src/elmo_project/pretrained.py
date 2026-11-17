"""Inference-only reader for the original bilm-tf HDF5 checkpoint.

TensorFlow gate order is i,g,f,o; forget bias is added at inference. CNN,
highway, projection and clipped LSTMP equations follow the published bilm-tf
format. Sentence state is reset explicitly (see README comparison caveats).
"""
import torch
from torch import nn
from .common import read_json
from .models import CharEncoder, mask_from_lengths, reverse_valid


class ReferenceCell(nn.Module):
    def __init__(self, kernel, bias, projection, cell_clip, projection_clip):
        super().__init__()
        self.register_buffer("kernel", kernel)
        self.register_buffer("bias", bias)
        self.register_buffer("projection", projection)
        self.cell_clip = cell_clip
        self.projection_clip = projection_clip

    def forward(self, x, lengths):
        batch, time, _ = x.shape
        cell = x.new_zeros(batch, self.projection.shape[0])
        hidden = x.new_zeros(batch, self.projection.shape[1])
        inputs = x @ self.kernel[:x.shape[-1]] + self.bias
        recurrent = self.kernel[x.shape[-1]:]
        output = []
        for t in range(time):
            i, g, f, o = (inputs[:, t] + hidden @ recurrent).chunk(4, -1)
            next_cell = torch.sigmoid(f + 1) * cell + torch.sigmoid(i) * torch.tanh(g)
            if self.cell_clip:
                next_cell = next_cell.clamp(-self.cell_clip, self.cell_clip)
            next_hidden = (torch.sigmoid(o) * torch.tanh(next_cell)) @ self.projection
            if self.projection_clip:
                next_hidden = next_hidden.clamp(-self.projection_clip, self.projection_clip)
            valid = (lengths > t)[:, None]
            cell = torch.where(valid, next_cell, cell)
            hidden = torch.where(valid, next_hidden, hidden)
            output.append(hidden * valid)
        return torch.stack(output, 1)


class PublicELMo(nn.Module):
    def __init__(self, options_path, weights_path):
        super().__init__()
        import h5py
        options = read_json(options_path)
        cnn, lstm = options["char_cnn"], options["lstm"]
        self.depth = lstm["n_layers"]
        self.width = 2 * lstm["projection_dim"]
        self.max_chars = cnn["max_characters_per_token"]
        self.residual = lstm.get("use_skip_connections", False)
        self.char = CharEncoder(cnn["embedding"]["dim"], cnn["filters"], lstm["projection_dim"], cnn.get("n_highway", 0), cnn["activation"])
        self.directions = nn.ModuleList()
        with h5py.File(weights_path, "r") as f, torch.no_grad():
            def tensor(key):
                return torch.from_numpy(f[key][...]).float()

            emb = tensor("char_embed")
            if emb.shape[0] != 261:
                raise ValueError(f"Expected 261 exported character rows, got {emb.shape}")
            self.char.embedding.weight.zero_()
            self.char.embedding.weight[1:].copy_(emb)
            for i, conv in enumerate(self.char.convs):
                conv.weight.copy_(tensor(f"CNN/W_cnn_{i}").squeeze(0).permute(2, 1, 0))
                conv.bias.copy_(tensor(f"CNN/b_cnn_{i}"))
            for i, highway in enumerate(self.char.highways):
                for attribute, source in (("gate", "carry"), ("transform", "transform")):
                    layer = getattr(highway, attribute)
                    layer.weight.copy_(tensor(f"CNN_high_{i}/W_{source}").T)
                    layer.bias.copy_(tensor(f"CNN_high_{i}/b_{source}"))
            if isinstance(self.char.projection, nn.Linear):
                self.char.projection.weight.copy_(tensor("CNN_proj/W_proj").T)
                self.char.projection.bias.copy_(tensor("CNN_proj/b_proj"))
            for direction in range(2):
                stack = nn.ModuleList()
                for layer in range(self.depth):
                    prefix = f"RNN_{direction}/RNN/MultiRNNCell/Cell{layer}/LSTMCell"
                    stack.append(ReferenceCell(tensor(prefix + "/W_0"), tensor(prefix + "/B"),
                                               tensor(prefix + "/W_P_0"), lstm.get("cell_clip"), lstm.get("proj_clip")))
                self.directions.append(stack)
        self.requires_grad_(False)
        self.eval()

    def forward(self, chars, lengths):
        x = self.char(chars)
        mask = mask_from_lengths(lengths, x.shape[1]).unsqueeze(-1)
        directions = []
        for direction, stack in enumerate(self.directions):
            hidden = reverse_valid(x, lengths) if direction else x
            values = []
            for i, cell in enumerate(stack):
                update = cell(hidden, lengths)
                hidden = (update + hidden if i and self.residual else update) * mask
                values.append(reverse_valid(hidden, lengths) if direction else hidden)
            directions.append(values)
        return torch.stack([torch.cat([x, x], -1)] + [torch.cat([directions[0][i], directions[1][i]], -1) for i in range(self.depth)], 2)
