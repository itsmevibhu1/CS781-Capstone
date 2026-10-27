def positive_weights(cfg):
    counts = TRAIN_COUNTS.astype(float)
    if not cfg['use_class_weights']:
        return None
    weights = ((len(DATA['train']) - counts) / np.maximum(counts, 1)) ** cfg['class_weight_power']
    # No artificial learning signal is created for statutes with no positive train examples.
    weights[counts == 0] = 1.0
    return torch.tensor(np.clip(weights, 1.0, cfg['pos_weight_cap']), dtype=torch.float32, device=DEVICE)


def multilabel_loss(logits, targets, weights, cfg):
    x = logits.float(); y = targets.float()
    smooth = cfg['label_smoothing']; y = y * (1 - smooth) + 0.5 * smooth
    if cfg['loss_name'] in ('bce', 'focal'):
        loss = F.binary_cross_entropy_with_logits(x, y, pos_weight=weights, reduction='none')
        if cfg['loss_name'] == 'focal':
            p = x.sigmoid(); pt = y * p + (1 - y) * (1 - p)
            loss = loss * (1 - pt).pow(cfg['focal_gamma'])
    elif cfg['loss_name'] == 'asymmetric':
        p = x.sigmoid(); negative = (1 - p + cfg['asl_clip']).clamp(max=1)
        positive_weight = weights if weights is not None else 1.0
        loss = -positive_weight * y * torch.log(p.clamp_min(1e-8)) * (1 - p).pow(cfg['asl_gamma_pos'])
        loss = loss - (1 - y) * torch.log(negative.clamp_min(1e-8)) * (1 - negative).pow(cfg['asl_gamma_neg'])
    else:
        raise ValueError('Unknown multi-label loss.')
    return loss.mean()


def distillation_loss(student, teacher, temperature):
    # Independent Bernoulli labels; softmax/KL would incorrectly make statutes mutually exclusive.
    return F.binary_cross_entropy_with_logits(student.float() / temperature,
        torch.sigmoid(teacher.float() / temperature)) * temperature ** 2


def save_error_analysis(output, keys, probabilities, policy, metrics):
    import csv
    predicted = decisions(probabilities, policy)
    rows = []
    for key, values, chosen in zip(keys, probabilities, predicted):
        gold = set(DATA['dev'][key]['label']); pred = {label for label, yes in zip(LABEL_IDS, chosen) if yes}
        if pred != gold:
            rows.append({'id': DATA['dev'][key]['id'], 'instance_key': key, 'gold': sorted(gold),
                         'pred': sorted(pred), 'missed': sorted(gold - pred), 'extra': sorted(pred - gold)})
    save_json(rows, output / 'dev_errors.json')
    with (output / 'per_label_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        columns = ['statute_id', 'train_support', 'support', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn']
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for label, values in metrics['per_label'].items():
            writer.writerow({'statute_id': label, **values})
    rare = [row for row in metrics['per_label'].values() if row['train_support'] <= 10 and row['support'] > 0]
    save_json({'rare_definition': 'train support <=10 and present in dev', 'rare_labels': len(rare),
               'rare_macro_f1': float(np.mean([row['f1'] for row in rare])) if rare else None,
               'mean_gold_labels': float(np.mean([len(DATA['dev'][key]['label']) for key in keys])),
               'mean_predicted_labels': metrics['mean_predicted_labels']}, output / 'rare_label_analysis.json')
    missed = sorted(metrics['per_label'].items(), key=lambda pair: pair[1]['fn'], reverse=True)[:20]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([label for label, row in missed], [row['fn'] for label, row in missed])
    ax.set(xlabel='Statute ID', ylabel='False negatives on DEV', title='Most missed statutes')
    fig.tight_layout(); fig.savefig(output / 'missed_statutes.png', dpi=130); plt.show(); plt.close(fig)
