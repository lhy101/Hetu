TP_VALUES=(8)
PP_VALUES=(2)

# 循环 TP 和 PP 的值
for TP in "${TP_VALUES[@]}"; do
  for PP in "${PP_VALUES[@]}"; do
    # 跳过 TP * PP 大于 16 的情况
    if (( TP * PP < 4 )); then
      continue
    fi
    if (( TP * PP > 16 )); then
      continue
    fi
    # 定义 EXP_FILE 的路径
    EXP_FILE="./experiments/hydraulis/tp${TP}_pp${PP}"
    # 调用现有脚本
    bash scripts/profile_all_seqlen.sh "$TP" "$PP" "$EXP_FILE"
  done
done

