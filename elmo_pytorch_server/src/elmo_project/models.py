"""Character biLM and supervised NER models, implemented with PyTorch."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def mask_from_lengths(lengths, size):
    return torch.arange(size, device=lengths.device)[None] < lengths[:, None]


def reverse_valid(x, lengths):
    positions = torch.arange(x.shape[1], device=x.device)[None].expand(x.shape[0], -1)
    indices = torch.where(positions < lengths[:, None], lengths[:, None] - 1 - positions, positions)
    return x.gather(1, indices.unsqueeze(-1).expand_as(x))


class Highway(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.transform = nn.Linear(size, size)
        self.gate = nn.Linear(size, size)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, x):
        gate = torch.sigmoid(self.gate(x))
        return gate * F.relu(self.transform(x)) + (1 - gate) * x


class CharEncoder(nn.Module):
    def __init__(self, char_dim, filters, projection, highways=2, activation="relu"):
        super().__init__()
        self.embedding = nn.Embedding(262, char_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(char_dim, count, width) for width, count in filters])
        size = sum(count for _, count in filters)
        self.highways = nn.ModuleList([Highway(size) for _ in range(highways)])
        self.projection = nn.Linear(size, projection) if size != projection else nn.Identity()
        self.activation = activation

    def forward(self, chars):
        shape = chars.shape[:-1]
        flat = chars.reshape(-1, chars.shape[-1])
        # Repeated word types share the same differentiable CNN result in a batch.
        unique, inverse = torch.unique(flat, dim=0, return_inverse=True)
        x = self.embedding(unique).transpose(1, 2)
        activation = torch.tanh if self.activation == "tanh" else F.relu
        x = torch.cat([activation(conv(x).amax(-1)) for conv in self.convs], -1)
        for highway in self.highways:
            x = highway(x)
        x = self.projection(x)[inverse].reshape(*shape, -1)
        return x * chars.ne(0).any(-1, keepdim=True)


class BiLM(nn.Module):
    def __init__(self, config, vocab_size, depth):
        super().__init__()
        self.depth, self.width = depth, config["projection"] * 2
        self.char = CharEncoder(config["char_dim"], config["filters"], config["projection"], config["highways"])
        self.directions = nn.ModuleList([nn.ModuleList([
            nn.LSTM(config["projection"], config["hidden"], batch_first=True, proj_size=config["projection"])
            for _ in range(depth)]) for _ in range(2)])
        self.output = nn.Linear(config["projection"], vocab_size)
        self.dropout = nn.Dropout(config["dropout"])
        # One forget bias per cell, not one per bias vector.
        for stack in self.directions:
            for lstm in stack:
                h = config["hidden"]
                with torch.no_grad():
                    lstm.bias_ih_l0[h:2*h].fill_(1)
                    lstm.bias_hh_l0.zero_()

    def forward(self, chars, lengths):
        x = self.char(chars)
        outputs = []
        mask = mask_from_lengths(lengths, x.shape[1]).unsqueeze(-1)
        for direction, stack in enumerate(self.directions):
            hidden = reverse_valid(x, lengths) if direction else x
            layers = []
            for index, lstm in enumerate(stack):
                packed = pack_padded_sequence(self.dropout(hidden), lengths.cpu(), batch_first=True, enforce_sorted=False)
                result, _ = lstm(packed)
                result, _ = pad_packed_sequence(result, batch_first=True, total_length=x.shape[1])
                hidden = (result + hidden if index else result) * mask
                layers.append(reverse_valid(hidden, lengths) if direction else hidden)
            outputs.append(layers)
        mixed = [torch.cat([x, x], -1)]
        mixed += [torch.cat([outputs[0][i], outputs[1][i]], -1) for i in range(self.depth)]
        return torch.stack(mixed, dim=2)  # [batch, time, depth+1, 2*projection]

    def loss(self, chars, targets, lengths, chunk_size=128):
        layers = self(chars, lengths)
        width = self.width // 2
        forward = layers[:, :-1, -1, :width]
        backward = layers[:, 1:, -1, width:]
        mask = mask_from_lengths(lengths - 1, chars.shape[1] - 1)
        counts = int(mask.sum())
        losses = []
        for hidden, target in ((forward[mask], targets[:, 1:][mask]), (backward[mask], targets[:, :-1][mask])):
            # Chunk the exact softmax to bound peak memory with a 50k vocabulary.
            sums = hidden.new_zeros((), dtype=torch.float32)
            for start in range(0, len(target), chunk_size):
                def chunk_loss(h, y):
                    return F.cross_entropy(self.output(h).float(), y, reduction="sum")
                h, y = hidden[start:start + chunk_size], target[start:start + chunk_size]
                if self.training and torch.is_grad_enabled():
                    sums = sums + checkpoint(chunk_loss, h, y, use_reentrant=False)
                else:
                    sums = sums + chunk_loss(h, y)
            losses.append(sums)
        return (losses[0] + losses[1]) / (2 * counts), losses, counts


class CRF(nn.Module):
    def __init__(self, tags):
        super().__init__()
        self.n = len(tags)
        self.start = nn.Parameter(torch.zeros(self.n))
        self.end = nn.Parameter(torch.zeros(self.n))
        self.transitions = nn.Parameter(torch.zeros(self.n, self.n))  # previous, next
        allowed = torch.ones(self.n, self.n, dtype=torch.bool)
        start_allowed = torch.ones(self.n, dtype=torch.bool)
        for j, tag in enumerate(tags):
            if tag.startswith("I-"):
                start_allowed[j] = False
                for i, previous in enumerate(tags):
                    allowed[i, j] = previous in ("B-" + tag[2:], tag)
        self.register_buffer("allowed", allowed)
        self.register_buffer("start_allowed", start_allowed)

    def potentials(self):
        return self.transitions.masked_fill(~self.allowed, -10000), self.start.masked_fill(~self.start_allowed, -10000)

    def log_partition(self, emissions, mask):
        trans, start = self.potentials()
        alpha = start + emissions[:, 0]
        for t in range(1, emissions.shape[1]):
            updated = torch.logsumexp(alpha[:, :, None] + trans + emissions[:, t, None, :], dim=1)
            alpha = torch.where(mask[:, t, None], updated, alpha)
        return torch.logsumexp(alpha + self.end, dim=1)

    def forward(self, emissions, tags, mask):
        trans, start = self.potentials()
        batch = torch.arange(len(tags), device=tags.device)
        value = start[tags[:, 0]] + emissions[batch, 0, tags[:, 0]]
        for t in range(1, tags.shape[1]):
            step = trans[tags[:, t - 1], tags[:, t]] + emissions[batch, t, tags[:, t]]
            value = value + step * mask[:, t]
        last = mask.long().sum(1) - 1
        value = value + self.end[tags[batch, last]]
        return (self.log_partition(emissions, mask) - value).mean()

    def decode(self, emissions, mask):
        trans, start = self.potentials()
        value = start + emissions[:, 0]
        history = []
        for t in range(1, emissions.shape[1]):
            candidate, source = (value[:, :, None] + trans).max(dim=1)
            value = torch.where(mask[:, t, None], candidate + emissions[:, t], value)
            history.append(source)
        ends = (value + self.end).argmax(1)
        result = []
        for row, length in enumerate(mask.sum(1).tolist()):
            tag = ends[row].item()
            path = [tag]
            for t in range(length - 2, -1, -1):
                tag = history[t][row, tag].item()
                path.append(tag)
            result.append(list(reversed(path)))
        return result


class LockedDropout(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.p = probability

    def forward(self, x):
        if not self.training or not self.p:
            return x
        mask = x.new_empty(x.shape[0], 1, x.shape[-1]).bernoulli_(1 - self.p) / (1 - self.p)
        return x * mask


class NER(nn.Module):
    def __init__(self, cfg, vectors, tags, feature_shape=None):
        super().__init__()
        self.words = nn.Embedding.from_pretrained(vectors, freeze=False, padding_idx=0)
        self.char = CharEncoder(cfg["char_dim"], [[3, cfg["char_filters"]]], cfg["char_filters"], highways=0)
        width = vectors.shape[1] + cfg["char_filters"]
        self.mix_l2 = cfg["mix_l2"]
        self.mix = nn.Parameter(torch.zeros(feature_shape[0])) if feature_shape else None
        self.gamma = nn.Parameter(torch.ones(())) if feature_shape else None
        if feature_shape:
            width += feature_shape[1]
        self.elmo_dropout = nn.Dropout(cfg["elmo_dropout"])
        self.dropout = LockedDropout(cfg["dropout"])
        self.layers = nn.ModuleList()
        for hidden in cfg["hidden"]:
            self.layers.append(nn.LSTM(width, hidden, batch_first=True, bidirectional=True))
            width = 2 * hidden
        self.classifier = nn.Linear(width, len(tags))
        self.crf = CRF(tags)

    def forward(self, words, chars, lengths, features=None):
        pieces = [self.words(words), self.char(chars)]
        if self.mix is not None:
            if features is None:
                raise ValueError("Contextual features missing")
            mixed = (features.float() * self.mix.softmax(0)[None, None, :, None]).sum(2) * self.gamma
            pieces.append(self.elmo_dropout(mixed))
        x = torch.cat(pieces, -1)
        for lstm in self.layers:
            packed = pack_padded_sequence(self.dropout(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
            x, _ = lstm(packed)
            x, _ = pad_packed_sequence(x, batch_first=True, total_length=words.shape[1])
        return self.classifier(x)

    def loss(self, emissions, tags, mask):
        result = self.crf(emissions.float(), tags, mask)
        if self.mix is not None:
            result = result + self.mix_l2 * self.mix.square().sum()
        return result
