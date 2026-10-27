"""Patch rsl_rl's OnPolicyRunner to number checkpoints by completed iterations (1-based).

Before: `current_learning_iteration = it` (0-based), so a 3000-iteration run saved its
final checkpoint as model_2999.pt.  After: `it + 1`, so the final file is model_3000.pt
and the stored "iter" matches the number of finished iterations.

Usage: python patch_rslrl_iter.py <path/to/on_policy_runner.py>
"""
import sys

path = sys.argv[1]
src = open(path).read()

old_assign = "            self.current_learning_iteration = it\n"
new_assign = (
    "            # 1-based count of completed iterations.  `it` is a 0-based loop index, so a\n"
    "            # 3000-iteration run used to end at model_2999.pt.\n"
    "            self.current_learning_iteration = it + 1\n"
)
assert src.count(old_assign) == 1, f"assign occurrences: {src.count(old_assign)}"
src = src.replace(old_assign, new_assign)

old_save = (
    'if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:\n'
    '                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore'
)
new_save = (
    'if (\n'
    '                self.logger.writer is not None\n'
    '                and self.current_learning_iteration % self.cfg["save_interval"] == 0\n'
    '            ):\n'
    '                self.save(\n'
    '                    os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")\n'
    '                )  # type: ignore'
)
assert src.count(old_save) == 1, f"save occurrences: {src.count(old_save)}"
src = src.replace(old_save, new_save)

open(path, "w").write(src)
print("patched:", path)
