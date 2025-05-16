MODEL_PARAMS = {
  '700M': dict(hidden_size=1536, glu_size=4096, n_heads=24, n_layers=24, lr_start=1.5*10e-3, lr_end=1*10e-3),
  '1.3B': dict(hidden_size=2048, glu_size=5460, n_heads=32, n_layers=24, lr_start=1.2*10e-3, lr_end=8*10e-4),
  '3B': dict(hidden_size=3200, glu_size=8640, n_heads=32, n_layers=26, lr_start=1.2*10e-3, lr_end=8*10e-4),
  '3.9B': dict(hidden_size=3200, glu_size=12800, n_heads=32, n_layers=26, lr_start=1.2*10e-3, lr_end=8*10e-4),
}

def get_model_params(model_size):
    if model_size not in MODEL_PARAMS: raise ValueError(f"Model size {model_size} not found in MODEL_PARAMS")
    hyperparams = {
        'BATCH_SIZE': 100_000_000,
        'NUM_TOKENS': 100_000_000_000,
        'SEQ_LEN': 2048,
        'WARMUP': 375,
        'ADAM_BETA': (0.9, 0.95)
    }
    return MODEL_PARAMS[model_size] | hyperparams

print(get_model_params('700M'))