# Final hand-off summary

The final contribution is a SAC locomotion pipeline for Unitree Go2 that moves
from a dedicated high-stair policy to a mixed-terrain policy while explicitly
monitoring catastrophic forgetting of the 23 cm capability.

## Key stages

1. **V48:** focused 21.5--23 cm curriculum, fixed 22/23 cm anchors, adaptive
   swing-clearance reward, and a deliberately small actor learning rate.
2. **V49:** rebalanced distribution (55% upstairs, 45% generalisation terrain),
   retaining a fixed 15% 23 cm anchor and selecting checkpoints through a
   common seven-case matrix.
3. **V50:** curriculum recovery with promotion based on progress/survival and a
   ceiling that avoids successful robots wrapping back to random low levels.

## Raw final numbers

| Case | V49 model 4980 | V49 model 5177 | V50 model 5099 |
|---|---:|---:|---:|
| Fixed upstairs 23 cm | 24 | 20 | 24 |
| Plane | 32 | 32 | 32 |
| Downstairs 18 cm | 28 | 29 | 26 |
| Uphill | 32 | 32 | 32 |
| Downhill | 32 | 32 | 32 |
| Boxes | 8 | 8 | 7 |
| Random rough | 26 | 27 | 26 |
| Six-terrain mean | 26.33 | 26.67 | 25.83 |
| Six-terrain minimum | 8 | 8 | 7 |

V49 model 5177 is the evaluator's recommendation because it satisfied the
minimum 16/32 retention threshold at 23 cm and achieved the highest six-terrain
mean. V50 is retained as the final curriculum-recovery result because it raised
the training curriculum level to 6.3861 while matching model 4980's 24/32 fixed
23 cm result. These serve different selection objectives; neither should be
described as universally superior.

Raw sources are the `summary.json` files under the V49 and V50 evaluation
folders. Most development used one training seed and one fixed evaluation seed;
these results do not establish sim-to-real robustness or cross-seed statistical
significance. The box terrain remains the weakest test case.

