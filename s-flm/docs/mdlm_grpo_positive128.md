# Positive128 coverage expansion

Entry scripts/run_grpo_positive128.py on server53432. Before launch remove only
verified unpacked migration/upload archives; keep model/data/results/backups.
Cleanup manifest /root/autodl-tmp/cleanup-20261025.json.

Retain exact original32 successful programs. Additional train questions are
hash-ordered, disjoint by ID and normalized text from validation and original32;
consider at most512 candidate questions. Base EMA FP32 generates8 candidates per
question at T=.3/NFE32. Evaluate all8 and retain first correct; skip questions with
no success until96 new successes. Stop with FAILED if budget is exhausted.
Revalidate original successes. Retry infrastructure evaluation failures up to3
attempts, never interpret them as model failure. Save candidate order, all candidate
programs, seeds, source hashes, collection statistics and selected128 before training.

Train scripts/train_grpo_positive128.py from Base EMA in a fresh process. Positive
loss = -.25 * mean over128 programs of masked-token mean logp. This preserves the
original success-term scale (.5 advantage with half the paired denominator).
Original32 programs retain their exact train and four held-out masks from the
positive32 mask_plan. New96 get deterministic fixed75% masks. FP32, LoRA, AdamW
LR3e-5, cap .2, microbatch2,200 full accumulated updates. No KL/clipping. Scores
at0/1/2/10/50/100/200, adapter snapshots50/100/200, final evaluation checkpoint,
frozen parameter hash check. Coverage and total compute increase together.

After fitting automatically compare32 vs128 on all632 existing development
validation questions,16 draws each at T=.3/.1 with matching seeds on A800. No
Base evaluation. This is development validation, not a new test set. Script
scripts/evaluate_grpo_positive128.py saves raw/recovery execution records,
infrastructure failures and question-bootstrap differences. No selection of
checkpoint based on validation. Pipeline COMPLETE/FAILED and STAGE track status;
initial collect-successes is data preparation, not an optimizer update.
