class LoRAProjection(nn.Module):
    def __init__(self, base, rank, alpha, multiplicative=False):
        super().__init__(); self.base = base; base.requires_grad_(False)
        self.multiplicative = multiplicative; self.scale = alpha / rank
        self.A = nn.Linear(base.out_features if multiplicative else base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5)); nn.init.zeros_(self.B.weight)
    def forward(self, x):
        wx = F.linear(x, self.base.weight, None)
        update = self.B(self.A(wx if self.multiplicative else x)) * self.scale
        return wx + update if self.base.bias is None else wx + update + self.base.bias


class Bottleneck(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__(); self.norm = nn.LayerNorm(dim); self.down = nn.Linear(dim, hidden); self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
    def forward(self, x):
        return x + self.up(F.gelu(self.down(self.norm(x))))


def load_generative_phase1(architecture):
    path = PHASE1_DIR / ('run_' + architecture) / 'final_model.pt'
    if not path.is_file():
        raise FileNotFoundError('Missing Phase 1 checkpoint: ' + str(path))
    state = torch_load(path); config = infer_phase1_config(architecture, state)
    model = MODEL_CLASSES[architecture](**config)
    model.load_state_dict(state, strict=True)
    del state; gc.collect()
    return model, config


class LegalSummarizer(nn.Module):
    def __init__(self, backbone, phase1_config, cfg):
        super().__init__(); self.backbone = backbone; self.cfg = cfg
        self.architecture = cfg['spec']['architecture']; self.phase1_config = phase1_config
        self.max_context = phase1_config.get('max_len', 4096)
        method = cfg['spec']['method']; self.selected_layers = None
        if method == 'kd':
            self.selected_layers = list(range(0, len(backbone.blocks), 2))
            backbone.blocks = nn.ModuleList([backbone.blocks[i] for i in self.selected_layers])
        backbone.requires_grad_(method in ('full', 'kd'))
        if method == 'partial':
            collections = [backbone.enc_layers, backbone.dec_layers] if self.architecture == 'encoder_decoder' else [backbone.blocks]
            for layers in collections:
                for layer in list(layers)[-cfg['partial_layers']:]: layer.requires_grad_(True)
            for name, module in backbone.named_modules():
                if name in ['ln_f', 'norm_f', 'ln_enc_f', 'ln_dec_f']: module.requires_grad_(True)
        if method in ('lora', 'lorma'):
            for name, child in list(backbone.named_modules()):
                if isinstance(child, nn.Linear) and any(part in name for part in ['attn.', 'self_attn.', 'cross_attn.']):
                    parent, leaf = name.rsplit('.', 1)
                    setattr(backbone.get_submodule(parent), leaf, LoRAProjection(child, cfg['rank'], cfg['alpha'], method == 'lorma'))
        dim = backbone.tok_emb.embedding_dim
        self.adapters = nn.ModuleDict()
        if method in ('houlsby', 'pfeiffer'):
            collections = [('enc', backbone.enc_layers), ('dec', backbone.dec_layers)] if self.architecture == 'encoder_decoder' else [('stack', backbone.blocks)]
            for prefix, layers in collections:
                for i in range(len(layers)):
                    # Houlsby placement: after attention and feed-forward residuals.
                    # Pfeiffer-style comparison: one feed-forward/output bottleneck per block.
                    sites = ['attn', 'ff'] if method == 'houlsby' else ['ff']
                    if prefix == 'dec' and method == 'houlsby': sites.append('cross')
                    for site in sites: self.adapters[f'{prefix}_{i}_{site}'] = Bottleneck(dim, cfg['bottleneck_dim'])
        self.soft_prompt = None
        if method == 'prompt':
            self.soft_prompt = nn.Parameter(torch.empty(cfg['prompt_tokens'], dim))
            nn.init.normal_(self.soft_prompt, std=0.02)
        if not any(p.requires_grad for p in self.parameters()):
            raise ValueError('No trainable parameters in this profile.')

    def adapt(self, x, name):
        return self.adapters[name](x) if name in self.adapters else x

    @staticmethod
    def attend(attn, query, memory=None, keep=None, bias=None, past=None, rope=None, static_kv=None):
        memory = query if memory is None else memory
        batch, length, dim = query.shape; heads = attn.n_head; head_dim = attn.d_head
        if static_kv is not None:
            q = attn.q_proj(query); kv_heads = heads
            k, v = static_kv
        elif hasattr(attn, 'qkv'):
            q, k, v = attn.qkv(query).split(dim, dim=-1)
            kv_heads = heads
        else:
            q = attn.q_proj(query); k = attn.k_proj(memory); v = attn.v_proj(memory)
            kv_heads = getattr(attn, 'n_kv_head', heads)
        q = q.view(batch, length, heads, head_dim).transpose(1, 2)
        if static_kv is None:
            k = k.view(batch, -1, kv_heads, head_dim).transpose(1, 2)
            v = v.view(batch, -1, kv_heads, head_dim).transpose(1, 2)
        if rope is not None: q, k = apply_rope(q, k, *(value.to(q.dtype) for value in rope))
        if past is not None: k = torch.cat([past[0], k], dim=2); v = torch.cat([past[1], v], dim=2)
        cache = (k, v)
        if kv_heads != heads:
            k = k.repeat_interleave(heads // kv_heads, dim=1); v = v.repeat_interleave(heads // kv_heads, dim=1)
        mask = keep
        if bias is not None:
            mask = bias if keep is None else bias.masked_fill(~keep, float('-inf'))
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        output = output.transpose(1, 2).reshape(batch, length, dim)
        projection = attn.proj if hasattr(attn, 'proj') else attn.o_proj
        return projection(output), cache

    def encode(self, source, source_mask):
        b = self.backbone; parts = []
        size = self.cfg['encoder_chunk_size'] or source.shape[1]
        for start in range(0, source.shape[1], size):
            ids = source[:, start:start + size]; keep = source_mask[:, start:start + size]
            x = b.tok_emb(ids)
            # A fully padded chunk for a shorter batch member must still have one finite attention key.
            safe = keep.clone(); safe[:, 0] = True
            bias = b.enc_rel_bias(ids.shape[1], ids.shape[1], ids.device)
            for i, layer in enumerate(b.enc_layers):
                def block(h, layer=layer, i=i, safe=safe, bias=bias):
                    norm = layer.ln1(h)
                    update, _ = self.attend(layer.attn, norm, keep=safe[:, None, None, :], bias=bias)
                    h = self.adapt(h + update, f'enc_{i}_attn')
                    return self.adapt(h + layer.ff(layer.ln2(h)), f'enc_{i}_ff')
                x = checkpoint(block, x, use_reentrant=False) if self.training and self.cfg['gradient_checkpointing'] else block(x)
            parts.append(b.ln_enc_f(x))
        return torch.cat(parts, dim=1), source_mask

    def hidden(self, ids, keep, prefix_lengths=None, past=None, encoder=None, use_cache=False):
        b = self.backbone; architecture = self.architecture
        offset = past[0][0].shape[2] if past else 0
        x = b.tok_emb(ids)
        if self.soft_prompt is not None and offset == 0:
            x = torch.cat([self.soft_prompt[None].expand(len(ids), -1, -1), x], dim=1)
            keep = torch.cat([torch.ones(len(ids), len(self.soft_prompt), dtype=torch.bool, device=ids.device), keep], dim=1)
            if prefix_lengths is not None: prefix_lengths = prefix_lengths + len(self.soft_prompt)
        length = x.shape[1]; total = offset + length
        qpos = torch.arange(offset, total, device=ids.device)
        kpos = torch.arange(total, device=ids.device)
        causal = kpos[None, :] <= qpos[:, None]
        mask = causal[None, None] & keep[:, None, None, :]
        if architecture == 'hybrid_prefixlm' and offset == 0:
            prompt = prefix_lengths[:, None, None]
            bidirectional = (qpos[None, :, None] < prompt) & (kpos[None, None, :] < prompt)
            mask = (causal[None] | bidirectional)[:, None] & keep[:, None, None, :]
        if architecture == 'decoder_only': x = x + b.pos_emb(qpos)[None]
        rotary = None
        if architecture in ('hybrid_prefixlm', 'lightweight'):
            cos, sin = build_rope_cache(total, b.tok_emb.embedding_dim // b.n_head, ids.device)
            rotary = (cos[:, :, offset:total, :], sin[:, :, offset:total, :])
        caches = []
        layers = b.dec_layers if architecture == 'encoder_decoder' else b.blocks
        bias = b.dec_rel_bias(total, total, ids.device)[:, :, offset:total, :] if architecture == 'encoder_decoder' else None
        for i, layer in enumerate(layers):
            old = past[i] if past else None
            def block(h, layer=layer, i=i, old=old):
                if architecture == 'encoder_decoder':
                    update, cache = self.attend(layer.self_attn, layer.ln1(h), keep=mask, bias=bias, past=old[:2] if old else None)
                    h = self.adapt(h + update, f'dec_{i}_attn')
                    update, cross_cache = self.attend(layer.cross_attn, layer.ln2(h), memory=encoder[0],
                                                      keep=encoder[1][:, None, None, :], static_kv=old[2:] if old else None)
                    if use_cache: cache = (*cache, *cross_cache)
                    h = self.adapt(h + update, f'dec_{i}_cross')
                    h = self.adapt(h + layer.ff(layer.ln3(h)), f'dec_{i}_ff')
                elif architecture == 'decoder_only':
                    update, cache = self.attend(layer.attn, layer.ln1(h), keep=mask, past=old)
                    h = self.adapt(h + update, f'stack_{i}_attn')
                    h = self.adapt(h + layer.mlp(layer.ln2(h)), f'stack_{i}_ff')
                else:
                    update, cache = self.attend(layer.attn, layer.norm1(h), keep=mask, past=old, rope=rotary)
                    h = self.adapt(h + update, f'stack_{i}_attn')
                    h = self.adapt(h + layer.ff(layer.norm2(h)), f'stack_{i}_ff')
                return (h, cache) if use_cache else h
            if self.training and self.cfg['gradient_checkpointing'] and not use_cache:
                x = checkpoint(block, x, use_reentrant=False)
            elif use_cache:
                x, cache = block(x); caches.append(cache)
            else:
                x = block(x)
        final_norm = b.ln_dec_f if architecture == 'encoder_decoder' else (b.ln_f if architecture == 'decoder_only' else b.norm_f)
        return final_norm(x), caches if use_cache else None

    def forward(self, batch):
        encoder = self.encode(batch['source'], batch['source_mask']) if self.architecture == 'encoder_decoder' else None
        hidden, _ = self.hidden(batch['input_ids'], batch['input_mask'], batch['prefix_lengths'], encoder=encoder)
        prompt = len(self.soft_prompt) if self.soft_prompt is not None else 0
        indices = batch['target_positions'] + prompt
        selected = hidden.gather(1, indices[:, :, None].expand(-1, -1, hidden.shape[-1]))
        return self.backbone.head(selected)


def make_model(cfg):
    backbone, phase1 = load_generative_phase1(cfg['spec']['architecture'])
    model = LegalSummarizer(backbone, phase1, cfg)
    return model, {'phase1_config': phase1, 'selected_phase1_layers': model.selected_layers,
                   'total_parameters': sum(p.numel() for p in model.parameters()),
                   'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}


def pack_batch(split, keys, cfg, targets=None):
    architecture = cfg['spec']['architecture']
    instruction = tok.encode(cfg['instruction'], add_special_tokens=False).ids
    separator = tok.encode(cfg['summary_prefix'], add_special_tokens=False).ids
    targets = targets or [(REFERENCES[split][key][:cfg['max_target_tokens'] - 1] + [EOS_ID]) for key in keys]
    longest_target = max(map(len, targets)); sequences = []; prefixes = []; sources = []; positions = []; coverage = []
    # Native relative-position encoder has a separate source budget. Decoder families share one context.
    soft = cfg['prompt_tokens'] if cfg['spec']['method'] == 'prompt' else 0
    context = cfg['context_limit']
    source_budget = cfg['max_source_tokens'] if architecture == 'encoder_decoder' else min(cfg['max_source_tokens'], context - soft - len(instruction) - len(separator) - longest_target - 2)
    if source_budget < 1: raise ValueError('Target/prompt leaves no source context; reduce lengths.')
    for key, target in zip(keys, targets):
        selected, ratio = select_source(DATA[split][key]['text'], TOKENS[split][key], source_budget, cfg['source_policy'])
        source = [BOS_ID] + selected + [EOS_ID]; sources.append(source); coverage.append(ratio)
        if architecture == 'encoder_decoder':
            sequence = [BOS_ID] + target[:-1]; prefix = 1
        else:
            prompt = [BOS_ID] + instruction + selected + separator
            prefix = len(prompt); sequence = prompt + target[:-1]
        sequences.append(sequence); prefixes.append(prefix)
        positions.append([prefix - 1 + j for j in range(len(target))])
    def pad(rows, value, dtype=torch.long):
        tensor = torch.full((len(rows), max(map(len, rows))), value, dtype=dtype)
        for i, row in enumerate(rows): tensor[i, :len(row)] = torch.tensor(row, dtype=dtype)
        return tensor.to(DEVICE)
    return {'keys': keys, 'targets_list': targets, 'input_ids': pad(sequences, PAD_ID),
            'input_mask': pad([[True] * len(row) for row in sequences], False, torch.bool),
            'prefix_lengths': torch.tensor(prefixes, device=DEVICE), 'source': pad(sources, PAD_ID),
            'source_mask': pad([[True] * len(row) for row in sources], False, torch.bool),
            'target_positions': pad(positions, 0), 'labels': pad(targets, -100), 'coverage': coverage}
