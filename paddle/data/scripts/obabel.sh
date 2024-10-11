#!/bin/bash

# 创建一个临时文件，删除重复的行
awk '!seen[$0]++' smiles_input.txt > smiles_cleaned.txt

# 遍历文件中的每一行
while IFS=',' read -r index smiles comment
do
    # 使用 OpenBabel 将 SMILES 转换为 SDF 文件
    obabel -:"$smiles" -O "${index}.sdf" --gen3D
done < smiles_cleaned.txt
