TP_VALUES=(8)
CP_VALUES=(2)

# 循环 TP 和 PP 的值
for TP in "${TP_VALUES[@]}"; do
  for CP in "${CP_VALUES[@]}"; do
    # 跳过 TP * PP 大于 16 的情况
    if (( TP * CP < 4 )); then
      continue
    fi
    if (( TP * CP > 16 )); then
      continue
    fi
    # 定义 EXP_FILE 的路径
    EXP_FILE="./experiments/hydraulis/tp${TP}_cp${CP}"
    # 调用现有脚本
    bash scripts/train_hetu_exp_cp.sh "$TP" "$CP" "$EXP_FILE"
  done
done

