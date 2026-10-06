# -*- coding: utf-8 -*-
"""
正则化回归 课后作业 第4题
用 Hitters 数据集对比 Ridge、Lasso、Elastic Net
数据来源: ISLR 包自带 Hitters (棒球运动员薪资数据)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

# ---------- 1. 读数据 ----------
DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv"
CSV = "Hitters.csv"
if not os.path.exists(CSV):
    print("本地没有 Hitters.csv，正在下载...")
    import urllib.request
    urllib.request.urlretrieve(DATA_URL, CSV)
    print("下载完成")

df = pd.read_csv(CSV)
print("原始数据 shape:", df.shape)

# Salary 有缺失，删掉
df = df.dropna(subset=["Salary"]).reset_index(drop=True)
print("去掉 Salary 缺失后 shape:", df.shape)

# 分类变量做 one-hot
cat_cols = ["League", "Division", "NewLeague"]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

y = df["Salary"].values
X = df.drop(columns=["Salary"]).values.astype(float)
feat_names = list(df.drop(columns=["Salary"]).columns)
print("特征数:", X.shape[1], " 样本数:", X.shape[0])

# ---------- 2. 划分 + 标准化 ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
# y 也标准化一下，方便系数对比（预测时再还原）
y_mean, y_std = y_train.mean(), y_train.std()
y_train_s = (y_train - y_mean) / y_std

# ---------- 3. 三个模型 cv=10 ----------
print("\n===== 交叉验证选超参 (cv=10) =====")

# Ridge: alpha 范围
ridge_alphas = np.logspace(-3, 3, 100)
ridge_cv = RidgeCV(alphas=ridge_alphas, cv=10)
ridge_cv.fit(X_train_s, y_train_s)
print("Ridge  最优 alpha =", round(ridge_cv.alpha_, 4))

# Lasso
lasso_alphas = np.logspace(-3, 1, 100)
lasso_cv = LassoCV(alphas=lasso_alphas, cv=10, max_iter=10000, random_state=42)
lasso_cv.fit(X_train_s, y_train_s)
print("Lasso  最优 alpha =", round(lasso_cv.alpha_, 4))

# Elastic Net: l1_ratio 也搜
en = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
    alphas=np.logspace(-3, 1, 100),
    cv=10, max_iter=10000, random_state=42,
)
en.fit(X_train_s, y_train_s)
print("ElasticNet 最优 alpha =", round(en.alpha_, 4),
      " l1_ratio =", en.l1_ratio_)

# ---------- 4. 测试集 RMSE ----------
def rmse(model):
    pred_s = model.predict(X_test_s)
    pred = pred_s * y_std + y_mean
    return np.sqrt(mean_squared_error(y_test, pred))

r_rmse = rmse(ridge_cv)
l_rmse = rmse(lasso_cv)
e_rmse = rmse(en)
print("\n===== 测试集 RMSE =====")
print("Ridge      :", round(r_rmse, 2))
print("Lasso      :", round(l_rmse, 2))
print("ElasticNet :", round(e_rmse, 2))

# ---------- 5. 非零系数个数 ----------
def nonzero(coef):
    return int(np.sum(np.abs(coef) > 1e-8))

print("\n===== 非零变量数 (总特征数 %d) =====" % X.shape[1])
print("Ridge      :", nonzero(ridge_cv.coef_), " (Ridge 不会严格为0)")
print("Lasso      :", nonzero(lasso_cv.coef_))
print("ElasticNet :", nonzero(en.coef_))

# 打印各模型系数(保留两位)
def show_coef(name, coef):
    print("\n--- %s 系数 ---" % name)
    items = sorted(zip(feat_names, coef), key=lambda x: -abs(x[1]))
    for fn, c in items:
        if abs(c) > 1e-8:
            print("  %-14s %+.3f" % (fn, c))

show_coef("Ridge", ridge_cv.coef_)
show_coef("Lasso", lasso_cv.coef_)
show_coef("ElasticNet", en.coef_)

# ---------- 6. 系数路径图 ----------
os.makedirs("figures", exist_ok=True)

def plot_path(model_class, alphas, title, fname, best_alpha, **kw):
    coefs = []
    for a in alphas:
        m = model_class(alpha=a, **kw)
        m.fit(X_train_s, y_train_s)
        coefs.append(m.coef_)
    coefs = np.array(coefs)
    plt.figure(figsize=(9, 5))
    for j in range(coefs.shape[1]):
        plt.plot(np.log10(alphas), coefs[:, j], label=feat_names[j], lw=1.2)
    plt.axvline(np.log10(best_alpha), color="k", ls="--", lw=1, label="CV最优")
    plt.xlabel("log10(alpha)")
    plt.ylabel("标准化系数")
    plt.title(title)
    plt.legend(fontsize=7, ncol=2, loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/" + fname, dpi=120)
    plt.close()
    print("已保存 figures/" + fname)

plot_path(Ridge, ridge_alphas, "Ridge 系数路径", "ridge_path.png", ridge_cv.alpha_, max_iter=10000)
plot_path(Lasso, lasso_alphas, "Lasso 系数路径", "lasso_path.png", lasso_cv.alpha_, max_iter=10000)
plot_path(ElasticNet, lasso_alphas, "ElasticNet 系数路径 (l1_ratio=%.1f)" % en.l1_ratio_,
          "en_path.png", en.alpha_, l1_ratio=en.l1_ratio_, max_iter=10000)

# ---------- 7. 1-SE 法则 (用 LassoCV 的 mse_path 近似) ----------
print("\n===== 1-SE 法则讨论 (Lasso) =====")
# LassoCV 存了 mse_path_ 和 alphas_
mse_means = lasso_cv.mse_path_.mean(axis=1)
mse_se = lasso_cv.mse_path_.std(axis=1) / np.sqrt(lasso_cv.mse_path_.shape[1])
best_idx = np.argmin(mse_means)
threshold = mse_means[best_idx] + mse_se[best_idx]
# 找最稀疏(alpha最大)且 mse <= threshold 的
candidates = np.where(mse_means <= threshold)[0]
onse_idx = candidates[np.argmax(lasso_cv.alphas_[candidates])]
onse_alpha = lasso_cv.alphas_[onse_idx]
print("最小 MSE 对应 alpha =", round(lasso_cv.alphas_[best_idx], 4))
print("1-SE 阈值 =", round(threshold, 4))
print("1-SE 法则选出 alpha =", round(onse_alpha, 4), "(更稀疏)")
onse_model = Lasso(alpha=onse_alpha, max_iter=10000).fit(X_train_s, y_train_s)
onse_pred = onse_model.predict(X_test_s) * y_std + y_mean
print("1-SE 模型测试 RMSE =", round(np.sqrt(mean_squared_error(y_test, onse_pred)), 2))
print("1-SE 模型非零变量数 =", nonzero(onse_model.coef_))

print("\n全部完成。图表在 figures/ 目录下。")
