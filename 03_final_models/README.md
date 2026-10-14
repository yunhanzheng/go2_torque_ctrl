# Selected final models

These checkpoints are the compact, non-duplicated model hand-off. All are
tracked using Git LFS.

| File | Purpose | SHA-256 |
|---|---|---|
| `V48_23cm/model_4778.pt` | Dedicated 23 cm staircase result | `b5355ed3eaefd17b7b606044594b720f9bdf291f9ae6c0f4b07462880d5dcfda` |
| `V49_generalization/model_4980.pt` | Balanced model retained as the source of V50 | `2b9d7892f925efcc413c7e438415c66ccc94708aecb155e78896d1caeb56f17a` |
| `V49_generalization/model_5177_recommended.pt` | V49 evaluator's recommended checkpoint | `d28f2873147c28902cdacdf3dfddaea2606b831d22b0510d10931d4d5b73f327` |
| `V50_curriculum_recovery/model_5099.pt` | Final curriculum-recovery checkpoint | `3291bb0fdf9063b1bb5ae3438fea3717a600aa14a08d2276bf30dfefa3d70227` |

Verify after cloning:

```bash
sha256sum 03_final_models/*/*.pt
```

These are PyTorch checkpoints. Only load them in a trusted Python environment.
V50 was produced by transferring actor and critic state from V49 model 4980
while reinitialising optimizers; see the V50 `train.py`.

