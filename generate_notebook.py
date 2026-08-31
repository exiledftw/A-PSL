import json
import codecs

# Read the original notebook to copy the base classes
with open('A-PSL_Phase1-2-3.ipynb', 'r', encoding='utf-8') as f:
    orig_nb = json.load(f)

# Keep the first 17 cells (Config, imports, dataset, collate, model classes)
new_cells = orig_nb['cells'][:17]

# Modify the Collate/DataLoader cell to add the 2000 sample limit
for i, cell in enumerate(new_cells):
    source = ''.join(cell['source'])
    if 'train_dataset, val_dataset = random_split' in source:
        new_source = source.replace(
            'n_total = len(full_dataset)',
            '# OPTUNA MODIFICATION: Shrink dataset to 2000 random clips\nimport random\nfull_dataset.samples = random.sample(full_dataset.samples, min(2000, len(full_dataset.samples)))\nn_total = len(full_dataset)'
        )
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        
# Add the Optuna installation cell
new_cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': ['!pip install -q optuna\n', 'import optuna\n']
})

# Add the Optuna Objective Function and Training Loop
optuna_code = '''
def objective(trial):
    # 1. Suggest Hyperparameters
    trial_lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    trial_dropout = trial.suggest_float("dropout", 0.1, 0.4)
    trial_layers = trial.suggest_int("num_encoder_layers", 2, 6, step=2)
    trial_warmup = trial.suggest_int("warmup_steps", 100, 1000)
    
    # 2. Update a copy of the config
    trial_config = CONFIG.copy()
    trial_config["LEARNING_RATE"] = trial_lr
    trial_config["DROPOUT"] = trial_dropout
    trial_config["NUM_ENCODER_LAYERS"] = trial_layers
    trial_config["WARMUP_STEPS"] = trial_warmup
    trial_config["MAX_EPOCHS"] = 3 # Fast epochs for tuning
    
    # 3. Initialize Model with trial config
    trial_model = SignLanguageTranslator(trial_config).to(device)
    
    # 4. Optimizer and Scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, trial_model.parameters()), 
        lr=trial_config["LEARNING_RATE"],
        weight_decay=trial_config["WEIGHT_DECAY"]
    )
    
    steps_per_epoch = len(train_loader) // trial_config["GRADIENT_ACCUMULATION_STEPS"]
    total_steps = steps_per_epoch * trial_config["MAX_EPOCHS"]
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=trial_config["WARMUP_STEPS"],
        num_training_steps=total_steps
    )
    
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler(enabled=trial_config["FP16"])
    
    # 5. Training Loop
    for epoch in range(trial_config["MAX_EPOCHS"]):
        trial_model.train()
        
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with autocast(enabled=trial_config["FP16"]):
                outputs = trial_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / trial_config["GRADIENT_ACCUMULATION_STEPS"]
                
            scaler.scale(loss).backward()
            
            if (step + 1) % trial_config["GRADIENT_ACCUMULATION_STEPS"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, trial_model.parameters()), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
        # 6. Validation Phase
        trial_model.eval()
        val_losses = []
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.no_grad():
                with autocast(enabled=trial_config["FP16"]):
                    outputs = trial_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_losses.append(outputs.loss.item())
            
        if len(val_losses) > 0:
            epoch_val_loss = sum(val_losses) / len(val_losses)
        else:
            epoch_val_loss = float("inf")
        
        # Report to Optuna for early pruning
        trial.report(epoch_val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return epoch_val_loss

print("Starting Optuna Study...")
study = optuna.create_study(direction="minimize")
# Run 10 trials
study.optimize(objective, n_trials=10)

print("\\n--- BEST TRIAL ---")
print(f"Best Val Loss: {study.best_trial.value}")
print("Best Hyperparameters:")
for key, value in study.best_trial.params.items():
    print(f"  {key}: {value}")
'''

new_cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [line + '\n' for line in optuna_code.split('\n')]
})

# Save the new notebook
new_nb = {
    'cells': new_cells,
    'metadata': orig_nb.get('metadata', {}),
    'nbformat': orig_nb.get('nbformat', 4),
    'nbformat_minor': orig_nb.get('nbformat_minor', 2)
}

with open('A-PSL_Optuna_Sweep.ipynb', 'w', encoding='utf-8') as f:
    json.dump(new_nb, f, indent=2)

print('Successfully created A-PSL_Optuna_Sweep.ipynb')
