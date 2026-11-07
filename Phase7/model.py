class ProjectionAdapter(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__(); self.base = base; base.requires_grad_(False)
        self.A = nn.Linear(base.out_features, rank, bias=False); self.B = nn.Linear(rank, base.out_features, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5)); nn.init.zeros_(self.B.weight)
    def forward(self, x):
        wx = F.linear(x, self.base.weight, None)
        y = wx + self.scale * self.B(self.A(wx))
        return y if self.base.bias is None else y + self.base.bias


class IA3Projection(nn.Module):
    def __init__(self, base, input_scale=False, fused_kv=False):
        super().__init__(); self.base = base; base.requires_grad_(False)
        self.input_scale = input_scale; self.fused_kv = fused_kv
        dimension = base.in_features if input_scale else (base.out_features * 2 // 3 if fused_kv else base.out_features)
        self.scale = nn.Parameter(torch.ones(dimension))
    def forward(self, x):
        if self.input_scale: return self.base(x * self.scale)
        y = self.base(x)
        scale = torch.cat([torch.ones(y.shape[-1] // 3, device=y.device, dtype=y.dtype), self.scale]) if self.fused_kv else self.scale
        return y * scale


class RetrievalFeatures(nn.Module):
    def __init__(self, backbone, architecture, cfg):
        super().__init__(); self.backbone = backbone; self.architecture = architecture; self.cfg = cfg
        self.soft_prompt = None
        if cfg['spec']['method'] == 'prompt':
            self.soft_prompt = nn.Parameter(torch.empty(cfg['prompt_tokens'], backbone.tok_emb.embedding_dim))
            nn.init.normal_(self.soft_prompt, std=0.02)
    def layer_call(self, layer, *args):
        if self.training and self.cfg['gradient_checkpointing'] and (args[0].requires_grad or any(p.requires_grad for p in layer.parameters())):
            return checkpoint(layer, *args, use_reentrant=False)
        return layer(*args)
    def forward(self, ids, mask, content):
        b = self.backbone; x = b.tok_emb(ids)
        if self.soft_prompt is not None:
            x = torch.cat([self.soft_prompt[None].expand(len(ids), -1, -1), x], dim=1)
            mask = torch.cat([torch.ones(len(ids), len(self.soft_prompt), dtype=torch.bool, device=ids.device), mask], dim=1)
            content = torch.cat([torch.zeros(len(ids), len(self.soft_prompt), dtype=torch.bool, device=ids.device), content], dim=1)
        length = x.shape[1]
        if self.architecture == 'decoder_only':
            if length > b.max_len: raise ValueError('Chunk/prompt exceeds Phase 1 positions.')
            x = x + b.pos_emb(torch.arange(length, device=ids.device))[None]
            for layer in b.blocks: x = self.layer_call(layer, x)
            x = b.ln_f(x)
        elif self.architecture == 'encoder_decoder':
            bias = b.enc_rel_bias(length, length, ids.device)
            for layer in b.enc_layers: x = self.layer_call(layer, x, bias, mask)
            x = b.ln_enc_f(x)
        else:
            cos, sin = build_rope_cache(length, b.d_model // b.n_head, ids.device)
            keep = mask[:, None, None, :].expand(-1, 1, length, -1)
            if self.architecture == 'lightweight':
                keep = keep & torch.ones(length, length, dtype=torch.bool, device=ids.device).tril()[None, None]
            for layer in b.blocks: x = self.layer_call(layer, x, keep, cos, sin)
            x = b.norm_f(x)
        weight = (mask & content).to(x.dtype)
        mean = (x * weight[:, :, None]).sum(1) / weight.sum(1, keepdim=True).clamp_min(1)
        final = x[torch.arange(len(ids), device=ids.device), mask.sum(1) - 1]
        return torch.cat([mean, final], dim=-1)


class LegalRetriever(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__(); self.cfg = cfg; self.features = RetrievalFeatures(backbone, cfg['spec']['architecture'], cfg)
        dimension = backbone.tok_emb.embedding_dim; out = cfg['embedding_dim']
        self.project = nn.Sequential(nn.LayerNorm(2 * dimension), nn.Linear(2 * dimension, out), nn.GELU(), nn.Dropout(cfg['dropout']))
        self.gate = nn.Linear(out, 1); self.mean_max = nn.Linear(2 * out, out)
        self.query_projection = nn.Linear(out, out, bias=False); self.candidate_projection = nn.Linear(out, out, bias=False)
        with torch.no_grad(): self.query_projection.weight.copy_(torch.eye(out)); self.candidate_projection.weight.copy_(torch.eye(out))
        self.candidate_projection.requires_grad_(not cfg['share_projection'])
        self.gate.requires_grad_(cfg['pooling'] == 'attention')
        self.mean_max.requires_grad_(cfg['pooling'] == 'mean_max')
    def forward(self, batch, role, return_chunks=False):
        features = []
        for start in range(0, len(batch['ids']), self.cfg['chunk_batch_size']):
            stop = start + self.cfg['chunk_batch_size']
            features.append(self.features(batch['ids'][start:stop], batch['mask'][start:stop], batch['content'][start:stop]))
        chunks = self.project(torch.cat(features)); documents = []
        for owner in range(len(batch['keys'])):
            h = chunks[batch['owners'] == owner]
            if self.cfg['pooling'] == 'attention': pooled = (h * torch.softmax(self.gate(h).squeeze(-1), dim=0)[:, None]).sum(0)
            elif self.cfg['pooling'] == 'mean': pooled = h.mean(0)
            elif self.cfg['pooling'] == 'max': pooled = h.max(0).values
            elif self.cfg['pooling'] == 'mean_max': pooled = self.mean_max(torch.cat([h.mean(0), h.max(0).values]))
            else: raise ValueError('Unknown pooling.')
            documents.append(pooled)
        projection = self.query_projection if self.cfg['share_projection'] or role == 'queries' else self.candidate_projection
        vectors = F.normalize(projection(torch.stack(documents)), dim=-1)
        if return_chunks: return vectors, F.normalize(projection(chunks), dim=-1), batch['owners']
        return vectors


def make_model(cfg):
    backbone, phase1, original_parameters = load_phase1(cfg['spec']['architecture'])
    soft = cfg['prompt_tokens'] if cfg['spec']['method'] == 'prompt' else 0
    if cfg['chunk_length'] + soft > phase1.get('max_len', 4096): raise ValueError('Chunk plus soft prompt exceeds Phase 1 positions.')
    method = cfg['spec']['method']; selected = None
    if method == 'kd':
        selected = list(range(0, len(backbone.blocks), 2)); backbone.blocks = nn.ModuleList([backbone.blocks[i] for i in selected])
    backbone.requires_grad_(method in ('full', 'kd'))
    targets = []
    if method == 'partial':
        layers = backbone.enc_layers if cfg['spec']['architecture'] == 'encoder_decoder' else backbone.blocks
        for layer in list(layers)[-cfg['partial_layers']:]: layer.requires_grad_(True)
        norm = backbone.ln_enc_f if cfg['spec']['architecture'] == 'encoder_decoder' else (backbone.ln_f if cfg['spec']['architecture'] == 'decoder_only' else backbone.norm_f)
        norm.requires_grad_(True)
    elif method == 'lorma':
        for name, child in list(backbone.named_modules()):
            if isinstance(child, nn.Linear) and '.attn.' in name:
                parent, leaf = name.rsplit('.', 1); setattr(backbone.get_submodule(parent), leaf, ProjectionAdapter(child, cfg['rank'], cfg['alpha'])); targets.append(name)
        assert targets
    elif method == 'ia3':
        for name, child in list(backbone.named_modules()):
            if not isinstance(child, nn.Linear): continue
            output = '.attn.' in name and name.rsplit('.', 1)[-1] in ('k_proj', 'v_proj', 'qkv')
            intermediate = name.endswith('.ff.w2') or name.endswith('.ff.2') or name.endswith('.mlp.2')
            if output or intermediate:
                parent, leaf = name.rsplit('.', 1)
                setattr(backbone.get_submodule(parent), leaf, IA3Projection(child, input_scale=intermediate, fused_kv=name.endswith('.qkv'))); targets.append(name)
        assert targets
    model = LegalRetriever(backbone, cfg)
    metadata = {'phase1_config': phase1, 'phase1_parameters': original_parameters, 'selected_phase1_layers': selected, 'adaptation_targets': targets,
                'total_parameters': sum(p.numel() for p in model.parameters()), 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    return model, metadata


def make_chunks(split, role, keys, cfg):
    all_chunks = []; owners = []; coverage = []
    for owner, key in enumerate(keys):
        ids = TOKENS[split][role][key]; size = cfg['chunk_length'] - 2
        starts = list(range(0, max(1, len(ids) - size + 1), size - cfg['chunk_overlap']))
        if ids and starts[-1] + size < len(ids): starts.append(max(0, len(ids) - size))
        starts = list(dict.fromkeys(starts))
        if cfg['max_chunks'] and len(starts) > cfg['max_chunks']:
            count = cfg['max_chunks']; policy = cfg['chunk_policy']
            if policy == 'head': starts = starts[:count]
            elif policy == 'tail': starts = starts[-count:]
            elif policy == 'head_tail': starts = starts[:(count + 1) // 2] + (starts[-(count // 2):] if count // 2 else [])
            else: starts = [starts[i] for i in np.linspace(0, len(starts) - 1, count).round().astype(int)]
        previous = 0; covered = 0
        for start in sorted(starts):
            stop = min(start + size, len(ids)); covered += max(0, stop - max(start, previous)); previous = max(previous, stop)
            all_chunks.append([BOS_ID] + (ids[start:stop] or [UNK_ID]) + [EOS_ID]); owners.append(owner)
        coverage.append(covered / len(ids) if ids else 1.0)
    length = max(map(len, all_chunks)); values = torch.full((len(all_chunks), length), PAD_ID, dtype=torch.long)
    mask = torch.zeros_like(values, dtype=torch.bool); content = torch.zeros_like(mask)
    for i, ids in enumerate(all_chunks):
        values[i, :len(ids)] = torch.tensor(ids); mask[i, :len(ids)] = True; content[i, 1:len(ids) - 1] = True
    return {'ids': values.to(DEVICE), 'mask': mask.to(DEVICE), 'content': content.to(DEVICE),
            'owners': torch.tensor(owners, device=DEVICE), 'keys': keys, 'coverage': coverage}
