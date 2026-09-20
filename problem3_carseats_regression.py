# -*- coding: utf-8 -*-
"""
第3题 Python 实操题：Carseats 数据集多元线性回归
==================================================
以 Sales（销售额）为响应变量，选取 Price、Income、Advertising 以及定性特征
ShelveLoc（货架位置），建立多元线性回归模型，并完成：
  1) 提取模型拟合报告，指出 ShelveLoc 的基准组；
  2) 解读 ShelveLoc[Good] 系数的实际商业含义；
  3) 计算各变量的 VIF，评估是否存在多重共线性风险。

运行方式：python problem3_carseats_regression.py
依赖：pandas, statsmodels, numpy
数据：Carseats.csv（与 R 中 ISLR 包的 Carseats 数据一致，400 个观测）
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 保证 Windows 控制台中文输出不乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 数据文件与脚本放在同一目录，确保任何工作目录下运行都能找到
DATA_PATH = Path(__file__).resolve().parent / "Carseats.csv"

# ---------------------------------------------------------------------------
# 1. 加载数据
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("=" * 78)
print("数据概览")
print("=" * 78)
print(f"样本量：{df.shape[0]}，变量数：{df.shape[1]}")
print(f"缺失值个数：{int(df.isnull().sum().sum())}")
print("ShelveLoc 各水平频数：")
print(df["ShelveLoc"].value_counts().to_string())
print()

# ---------------------------------------------------------------------------
# 2. 建立多元线性回归模型
#    Sales ~ Price + Income + Advertising + ShelveLoc(定性)
# ---------------------------------------------------------------------------
# 方式 A：使用 statsmodels 公式接口，C(ShelveLoc) 采用 Treatment 编码
model = ols("Sales ~ Price + Income + Advertising + C(ShelveLoc)", data=df).fit()

print("=" * 78)
print("模型拟合报告（statsmodels OLS Summary）")
print("=" * 78)
print(model.summary())
print()

# ---------------------------------------------------------------------------
# 3. 识别 ShelveLoc 的基准组
# ---------------------------------------------------------------------------
print("=" * 78)
print("问题1：ShelveLoc 的基准组是什么？")
print("=" * 78)
# Treatment（参照）编码下，第一个水平（按字母序）作为基准组被吸收进截距
shelve_levels = sorted(df["ShelveLoc"].unique())
print(f"ShelveLoc 的三个水平（按字母序）：{shelve_levels}")
print(f"基准组（reference / baseline）＝ {shelve_levels[0]}")
print("即：模型中以 Bad 为基准，报告中的 C(ShelveLoc)[T.Good] 与",
      "C(ShelveLoc)[T.Medium] 分别表示 Good、Medium 相对 Bad 的效应。")
print()

# ---------------------------------------------------------------------------
# 4. 解读 ShelveLoc[Good] 系数
# ---------------------------------------------------------------------------
print("=" * 78)
print("问题2：ShelveLoc[Good] 系数的实际商业含义")
print("=" * 78)
coef_good = model.params["C(ShelveLoc)[T.Good]"]
p_good = model.pvalues["C(ShelveLoc)[T.Good]"]
print(f"ShelveLoc[Good] 系数 = {coef_good:.4f}（p 值 = {p_good:.4g}）")
print(f"商业含义：在 Price、Income、Advertising 保持不变的前提下，"
      f"货架位置为 Good 的店")
print(f"铺平均比货架位置为 Bad 的店铺多销售 {coef_good:.4f} 千单位（即约 "
      f"{coef_good * 1000:.0f} 单位）商品。")
print(f"若系数显著（p 值 {p_good:.4g} < 0.05），说明“好货架位置”对销售额有",
      "统计上显著的提升作用。")
print()

# ---------------------------------------------------------------------------
# 5. 计算 VIF，评估多重共线性
# ---------------------------------------------------------------------------
print("=" * 78)
print("问题3：各变量的 VIF（方差膨胀因子）")
print("=" * 78)
# 构造设计矩阵：定性变量 ShelveLoc 做哑变量编码（drop_first=True，
# 与回归模型保持一致的基准组 Bad）
X = pd.get_dummies(df[["Price", "Income", "Advertising", "ShelveLoc"]],
                   columns=["ShelveLoc"], drop_first=True, dtype=float)
X = sm.add_constant(X)  # 加入截距列（VIF 计算时也包含截距）

vif_data = pd.DataFrame({
    "Variable": X.columns,
    "VIF": [variance_inflation_factor(X.values, i)
            for i in range(X.shape[1])],
})
print(vif_data.round(4).to_string(index=False))
print()
print("判断标准：VIF < 5 为低共线性；5 ≤ VIF < 10 为中等共线性；",
      "VIF ≥ 10 通常认为存在严重多重共线性。")
print()

# 为分类变量整体计算广义 VIF（整个哑变量组视为一个变量）
def group_vif(df_full, cols_group):
    """把一组哑变量整体作为一次回归的被解释变量，R^2 换算成广义 VIF。"""
    y_group = df_full[cols_group]
    x_others = df_full.drop(columns=cols_group)
    r2 = ols(f"y ~ {' + '.join(x_others.columns)}",
             data=pd.concat([y_group.assign(y=y_group.iloc[:, 0]),
                             x_others], axis=1)).fit().rsquared
    return 1.0 / (1.0 - r2)

X_no_const = X.drop(columns="const")
shelve_dummies = [c for c in X_no_const.columns if c.startswith("ShelveLoc_")]
print(f"ShelveLoc 哑变量组（{shelve_dummies}）的整体广义 VIF："
      f"{group_vif(X_no_const, shelve_dummies):.4f}")
print()

# ---------------------------------------------------------------------------
# 6. 结论小结
# ---------------------------------------------------------------------------
print("=" * 78)
print("结论小结")
print("=" * 78)
print(f"· 模型整体 R² = {model.rsquared:.4f}，调整后 R² = {model.rsquared_adj:.4f}")
max_vif = vif_data.loc[vif_data["Variable"] != "const", "VIF"].max()
print(f"· 所有解释变量的 VIF 均 < {max_vif:.2f}，"
      f"{'未发现明显的多重共线性风险。' if max_vif < 5 else '存在一定程度的多重共线性，需关注。'}")
print(f"· ShelveLoc 基准组为 Bad；Good 相对 Bad 的销售额增量 = {coef_good:.4f} 千单位。")
