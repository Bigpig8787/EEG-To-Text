#!/usr/bin/env bash
# run_paper_best_200hz.sh 的進度快照（印一次就結束）。固定畫面每 120 秒更新：
#   watch -n 120 bash scripts/watch_paper_best_200hz.sh
# 參數 1 = nohup 的輸出檔（預設 logs/ann_paper_best_200hz.out）。

cd "$(dirname "$0")/.." || exit 1
OUT=${1:-logs/ann_paper_best_200hz.out}
PRE=checkpoints/pretrain_paper_best_200hz
FT=checkpoints/paper_best_200hz
TAG=_paperbest200hz
PRE_EPOCHS=50

echo "== ANN paper-best 200Hz   $(date '+%F %T') =="
[[ -f $OUT ]] || { echo "找不到 $OUT（還沒開跑？）"; exit 0; }

# ── 行程 ──
cur=$(pgrep -af "python.*(train_pretrain|train_multiview|eval_multiview)\.py" | head -1)
if [[ -n $cur ]]; then
    pid=${cur%% *}
    echo "行程   RUNNING  pid $pid  已跑 $(ps -o etime= -p "$pid" | tr -d ' ')"
    echo "       $(echo "$cur" | cut -d' ' -f2- | grep -oE '(train_pretrain|train_multiview|eval_multiview)\.py.*' | cut -c1-90)"
elif pgrep -f run_paper_best_200hz.sh >/dev/null; then
    echo "行程   腳本在跑，正在切換階段"
else
    echo "行程   STOPPED（看下面的錯誤或最後幾行）"
fi

# ── 階段 ──
has() { compgen -G "$1" >/dev/null && echo "✅" || echo "⬜"; }
n_eval=$(compgen -G "results/*${TAG}*_metrics.json" | wc -l)
echo "階段   pretrain $(has "$PRE/encoder_best.pt")  STEP1 $(has "$FT/last/*${TAG}_s1.pt")  STEP2 $(has "$FT/best/*${TAG}_merged.pt")  eval ${n_eval}/4"

# 只看檔尾，tqdm 進度條會讓檔案很大
TXT=$(tail -c 400000 "$OUT" | tr '\r' '\n')

# ── pretrain 的 epoch 與預估 ──
pre_lines=$(grep -aE '^Epoch [0-9]+/[0-9]+ \| train' "$OUT" | tr '\r' '\n' | grep -aE '^Epoch [0-9]+/[0-9]+ \| train')
if [[ -n $pre_lines && ! -f $PRE/encoder_best.pt ]]; then
    done_ep=$(echo "$pre_lines" | wc -l)
    # nohup 會在第一行塞 "nohup: ignoring input"，所以抓第一個時間戳而不是 head -1
    start=$(date -d "$(grep -m1 -aoE '^\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:]{8}\]' "$OUT" | tr -d '[]')" +%s 2>/dev/null)
    if [[ -n $start ]]; then
        el=$(( $(date +%s) - start )); per=$(( el / done_ep )); left=$(( per * (PRE_EPOCHS - done_ep) ))
        echo "pretrain $done_ep/$PRE_EPOCHS epoch，$(awk -v p=$per -v l=$left 'BEGIN{printf "每 epoch 約 %.1f h，剩約 %.1f h", p/3600, l/3600}')（預計 $(date -d "@$(( $(date +%s) + left ))" '+%m-%d %H:%M')）"
    fi
    echo "$pre_lines" | tail -3 | sed 's/^/  /'
fi

# ── finetune 的 epoch ──
# pretrain 也印 "saved best"，所以 STEP 1 開始前不看這段
ft=""
if grep -qaE 'STEP 1 [^ ]+ encoder' "$OUT"; then
    # 從 STEP 1 標題之後才算（標題可能黏在 tqdm 那行後面，先切開）。
    # 不直接寫標題裡的「—」：非 UTF-8 locale 下 sed/grep 對不到。
    hdr='STEP [12] [^ ]* \(encoder\|LoRA\|full\)'
    sub=$(echo "$TXT" | sed "s/.*\($hdr\)/\1/" | sed -n '/^STEP 1 [^ ]* encoder/,$p')
    [[ -z $sub ]] && sub=$TXT
    ft=$(echo "$sub" | grep -aE '^STEP [12] |^Epoch [0-9]+/[0-9]+$|Loss: |saved best|no improvement|Early stopping|Merged checkpoint')
fi
[[ -n $ft ]] && { echo "finetune（最近）"; echo "$ft" | tail -8 | sed 's/^/  /'; }

# ── eval ──
ev=$(echo "$TXT" | grep -aE '^tf=|bleu-4:|rouge1_fmeasure:|^  wer:')
[[ -n $ev ]] && { echo "eval（最近）"; echo "$ev" | tail -8 | sed 's/^/  /'; }

# ── 目前的進度條 ──
bar=$(echo "$TXT" | grep -aE '[0-9]+/[0-9]+ \[[0-9:]+<' | tail -1)
[[ -n $bar ]] && echo "進度   ${bar:0:110}"

# ── 錯誤 ──
err=$(echo "$TXT" | grep -aE 'Traceback|Error|OutOfMemory|Killed|FileNotFound' | tail -3)
[[ -n $err ]] && { echo "⚠ 錯誤"; echo "$err" | sed 's/^/  /'; }

# ── GPU ──
echo "GPU"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader \
    | sed 's/^/  cuda:/'
