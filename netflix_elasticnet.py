# -*- coding: utf-8 -*-
"""
SVD 互动课件 课后作业
用 Elastic Net 对 Kaggle Netflix 数据做分析
数据: Netflix Movies and TV Shows (netflix_titles.csv)
目标: 预测影视作品的发行年份 release_year
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# ---------- 1. 读数据 ----------
DATA_URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-04-20/netflix_titles.csv"
CSV = "netflix_titles.csv"
if not os.path.exists(CSV):
    print("本地没有数据，正在下载 Netflix titles...")
    import urllib.request
    urllib.request.urlretrieve(DATA_URL, CSV)
    print("下载完成")

df = pd.read_csv(CSV)
print("原始数据 shape:", df.shape)
print("列名:", list(df.columns))

# ---------- 2. 数据清洗 ----------
# 去掉 release_year 缺失（其实没有缺失，但保险起见）
df = df.dropna(subset=["release_year"]).reset_index(drop=True)
df["release_year"] = df["release_year"].astype(int)

# 填充缺失
df["rating"] = df["rating"].fillna("Unknown")
df["country"] = df["country"].fillna("Unknown")
df["director"] = df["director"].fillna("")
df["cast"] = df["cast"].fillna("")
df["duration"] = df["duration"].fillna("0 min")

print("清洗后 shape:", df.shape)
print("发行年份范围: %d ~ %d" % (df["release_year"].min(), df["release_year"].max()))
print("类型分布:\n", df["type"].value_counts())

# ---------- 3. 特征工程 ----------
features = pd.DataFrame(index=df.index)

# 3.1 type: Movie / TV Show
features["is_movie"] = (df["type"] == "Movie").astype(int)

# 3.2 rating one-hot (取出现次数>=50的)
rating_counts = df["rating"].value_counts()
top_ratings = rating_counts[rating_counts >= 50].index.tolist()
for r in top_ratings:
    features["rating_" + r] = (df["rating"] == r).astype(int)

# 3.3 country: 取 top 15 国家, 其余 Other
def first_country(s):
    if pd.isna(s) or s == "Unknown":
        return "Unknown"
    return s.split(",")[0].strip()

df["main_country"] = df["country"].apply(first_country)
top_countries = df["main_country"].value_counts().head(15).index.tolist()
for c in top_countries:
    features["country_" + c] = (df["main_country"] == c).astype(int)
features["country_Other"] = (~df["main_country"].isin(top_countries)).astype(int)

# 3.4 listed_in (流派): multi-hot, 取出现>=100的流派
all_genres = []
for s in df["listed_in"].dropna():
    all_genres.extend([g.strip() for g in s.split(",")])
genre_counts = pd.Series(all_genres).value_counts()
top_genres = genre_counts[genre_counts >= 100].index.tolist()
print("\n纳入特征的流派数:", len(top_genres))
for g in top_genres:
    col = "genre_" + re.sub(r"[^a-zA-Z0-9]", "_", g)
    features[col] = df["listed_in"].fillna("").apply(lambda s: 1 if g in s else 0)

# 3.5 duration: 解析为数值
def parse_duration(s):
    s = str(s)
    if "min" in s:
        return int(re.sub(r"[^0-9]", "", s))
    elif "Season" in s:
        # 季数 * 10 近似换算, 只是为了和分钟同量级
        return int(re.sub(r"[^0-9]", "", s)) * 10
    return 0

features["duration_num"] = df["duration"].apply(parse_duration)

# 3.6 衍生特征
features["has_director"] = (df["director"] != "").astype(int)
features["cast_count"] = df["cast"].apply(lambda s: len(s.split(",")) if s else 0)
features["desc_len"] = df["description"].fillna("").apply(len)

# date_added 提取年份
def parse_year(s):
    try:
        return int(str(s).split(",")[-1].strip())
    except:
        return 0

features["added_year"] = df["date_added"].apply(parse_year)

print("最终特征数:", features.shape[1])

# ---------- 4. 划分 + 标准化 ----------
X = features.values.astype(float)
y = df["release_year"].values.astype(float)
feat_names = list(features.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
y_mean, y_std = y_train.mean(), y_train.std()
y_train_s = (y_train - y_mean) / y_std

print("训练集:", X_train_s.shape, " 测试集:", X_test_s.shape)

# ---------- 5. 三个模型 cv=10 ----------
print("\n===== 交叉验证 (cv=10) =====")

ridge_alphas = np.logspace(-3, 3, 100)
ridge_cv = RidgeCV(alphas=ridge_alphas, cv=10)
ridge_cv.fit(X_train_s, y_train_s)
print("Ridge  最优 alpha =", round(ridge_cv.alpha_, 4))

lasso_alphas = np.logspace(-4, 0, 100)
lasso_cv = LassoCV(alphas=lasso_alphas, cv=10, max_iter=20000, random_state=42)
lasso_cv.fit(X_train_s, y_train_s)
print("Lasso  最优 alpha =", round(lasso_cv.alpha_, 4))

en = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
    alphas=np.logspace(-4, 0, 100),
    cv=10, max_iter=20000, random_state=42,
)
en.fit(X_train_s, y_train_s)
print("ElasticNet 最优 alpha =", round(en.alpha_, 4),
      " l1_ratio =", en.l1_ratio_)

# ---------- 6. 测试集表现 ----------
def metrics(model):
    pred_s = model.predict(X_test_s)
    pred = pred_s * y_std + y_mean
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    return rmse, r2

r_rmse, r_r2 = metrics(ridge_cv)
l_rmse, l_r2 = metrics(lasso_cv)
e_rmse, e_r2 = metrics(en)

print("\n===== 测试集表现 =====")
print("%-12s %-10s %-8s %s" % ("模型", "RMSE", "R²", "非零变量数"))
print("%-12s %-10.2f %-8.4f %d" % ("Ridge", r_rmse, r_r2, np.sum(np.abs(ridge_cv.coef_) > 1e-8)))
print("%-12s %-10.2f %-8.4f %d" % ("Lasso", l_rmse, l_r2, np.sum(np.abs(lasso_cv.coef_) > 1e-8)))
print("%-12s %-10.2f %-8.4f %d" % ("ElasticNet", e_rmse, e_r2, np.sum(np.abs(en.coef_) > 1e-8)))

# ---------- 7. Elastic Net 系数解读 ----------
print("\n===== Elastic Net 非零系数 (按绝对值排序) =====")
items = sorted(zip(feat_names, en.coef_), key=lambda x: -abs(x[1]))
for fn, c in items:
    if abs(c) > 1e-8:
        print("  %-22s %+.4f" % (fn, c))

# ---------- 8. 系数路径图 ----------
os.makedirs("figures", exist_ok=True)

def plot_path(model_class, alphas, title, fname, best_alpha, **kw):
    coefs = []
    for a in alphas:
        m = model_class(alpha=a, **kw)
        m.fit(X_train_s, y_train_s)
        coefs.append(m.coef_)
    coefs = np.array(coefs)
    plt.figure(figsize=(10, 6))
    # 只画 top 15 重要的特征(按最终系数绝对值), 避免图例太乱
    top_idx = np.argsort(-np.abs(coefs[-1]))[:15]
    for j in top_idx:
        plt.plot(np.log10(alphas), coefs[:, j], label=feat_names[j], lw=1.3)
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

plot_path(Ridge, ridge_alphas, "Ridge 系数路径 (Netflix)", "netflix_ridge_path.png", ridge_cv.alpha_, max_iter=20000)
plot_path(Lasso, lasso_alphas, "Lasso 系数路径 (Netflix)", "netflix_lasso_path.png", lasso_cv.alpha_, max_iter=20000)
plot_path(ElasticNet, lasso_alphas, "ElasticNet 系数路径 (Netflix, l1_ratio=%.1f)" % en.l1_ratio_,
          "netflix_en_path.png", en.alpha_, l1_ratio=en.l1_ratio_, max_iter=20000)

# ---------- 9. Elastic Net 特征重要性条形图 ----------
top_n = 20
top_items = [x for x in items if abs(x[1]) > 1e-8][:top_n]
names = [x[0] for x in top_items][::-1]
vals = [x[1] for x in top_items][::-1]
colors = ["#2563eb" if v >= 0 else "#dc2626" for v in vals]
plt.figure(figsize=(9, 7))
plt.barh(range(len(names)), vals, color=colors)
plt.yticks(range(len(names)), names, fontsize=9)
plt.xlabel("标准化系数")
plt.title("Elastic Net 特征重要性 Top %d (Netflix)" % top_n)
plt.axvline(0, color="k", lw=0.8)
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/netflix_en_importance.png", dpi=120)
plt.close()
print("已保存 figures/netflix_en_importance.png")

# ---------- 10. 1-SE 法则 (ElasticNet) ----------
print("\n===== 1-SE 法则 (ElasticNet, l1_ratio=%.1f) =====" % en.l1_ratio_)
# ElasticNetCV 的 mse_path_ 形状是 (n_l1_ratio, n_alphas, n_folds)
# 找到最优 l1_ratio 对应的那一层
li_idx = list(en.l1_ratio_grid_).index(en.l1_ratio_) if hasattr(en, "l1_ratio_grid_") else 0
mse_path = en.mse_path_[li_idx] if en.mse_path_.ndim == 3 else en.mse_path_
mse_means = mse_path.mean(axis=1)
mse_se = mse_path.std(axis=1) / np.sqrt(mse_path.shape[1])
best_idx = np.argmin(mse_means)
threshold = mse_means[best_idx] + mse_se[best_idx]
candidates = np.where(mse_means <= threshold)[0]
onse_idx = candidates[np.argmax(en.alphas_[candidates])]
onse_alpha = en.alphas_[onse_idx]
print("最小 MSE alpha =", round(en.alphas_[best_idx], 4))
print("1-SE 阈值 =", round(threshold, 4))
print("1-SE 选出 alpha =", round(onse_alpha, 4))
onse_model = ElasticNet(alpha=onse_alpha, l1_ratio=en.l1_ratio_, max_iter=20000).fit(X_train_s, y_train_s)
onse_pred = onse_model.predict(X_test_s) * y_std + y_mean
onse_rmse = np.sqrt(mean_squared_error(y_test, onse_pred))
onse_r2 = r2_score(y_test, onse_pred)
print("1-SE 模型测试 RMSE = %.2f, R² = %.4f" % (onse_rmse, onse_r2))
print("1-SE 模型非零变量数 =", np.sum(np.abs(onse_model.coef_) > 1e-8))

print("\n全部完成。图表在 figures/ 目录下。")
