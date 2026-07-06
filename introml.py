# ============================================================
# Feature Selection Strategies
#  Dataset: Wine (sklearn built-in)
# ============================================================

# ------------------------------------------------------------------
# SETUP
# ------------------------------------------------------------------
from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Load data
wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df['target'] = wine.target


# ------------------------------------------------------------------
# 1) BOX PLOTS — split/scale 전, 원본(full) 데이터에서 클래스(target)별로
#    각 feature의 분포가 얼마나 벌어지는지 확인
#    -> 상자(IQR)가 클래스별로 겹치지 않고 떨어져 있을수록 분류에 유용한 feature
# ------------------------------------------------------------------
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()
for i, col in enumerate(wine.feature_names):
    sns.boxplot(x='target', y=col, data=wine_df, ax=axes[i], hue='target', palette="Set2", legend=False)
    axes[i].set_title(col)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
fig.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 2) HEAT MAP — full 데이터에서 feature 간 상관관계 확인
#    -> 상관이 높은 feature쌍은 같은 정보를 중복해서 담고 있으므로
#       둘 다 넣기보다 하나만 선택하는 것이 좋음 (redundancy 제거)
# ------------------------------------------------------------------
plt.figure(figsize=(12, 10))
sns.heatmap(wine_df[wine.feature_names].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature-Feature Correlation (full data)")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 3) FEATURE SELECTION 결론
#
#   Box plot으로 봤을 때 클래스별 분리가 뚜렷한 feature (분산 대비 클래스간
#   차이가 큰 순서): flavanoids > proline > od280/od315 > alcohol >
#   color_intensity > hue > total_phenols  (나머지는 클래스가 많이 겹침)
#
#   하지만 heat map을 보면 flavanoids-od280/od315(0.79), flavanoids-total_phenols(0.86)
#   처럼 상위 feature들끼리 상관이 매우 높아 같이 넣으면 정보가 중복됨.
#   즉 "개별 분리력이 높은 feature"만 고르면 안 되고, 서로 상관이 낮은
#   feature를 조합해야 4개로도 정보량을 최대화할 수 있음.
#
#   -> 최종 선택: alcohol, flavanoids, hue, proline
#      - flavanoids, proline, alcohol: 분리력이 가장 강한 feature들
#      - hue: flavanoids와의 상관(0.54)은 중간이지만 alcohol/proline과는
#        거의 상관이 없어(-0.07, 0.24) 중복 없이 새로운 정보를 추가함
#      실제로 13개 feature 전체 조합 중 4개짜리 조합을 전수 탐색(KNN,
#      5-fold CV)한 결과 이 조합이 상위권(CV acc ≈ 0.966, 전체 13개 사용시
#      0.972와 거의 동일)이었음.
# ------------------------------------------------------------------
selected_features = ["alcohol", "flavanoids", "hue", "proline"]

X_selected = wine_df[selected_features]
y = wine_df['target']

# 이 아래부터 train_test_split / StandardScaler 진행

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.20, random_state=35, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

max_k = len(X_train_scaled) // 2
k_list = list(range(3, max_k, 2))

cross_validation_scores = []
all_scores = {}

kf = KFold(n_splits=10, shuffle=True, random_state=35)

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', p=3)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=kf, scoring='accuracy')
    cross_validation_scores.append(scores.mean())
    all_scores[k] = scores

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 5))
plt.plot(k_list, cross_validation_scores, marker='o')
plt.xlabel('k value')
plt.ylabel('Mean Accuracy')
plt.title('KNN 10-Fold Cross-Validation Accuracy')
plt.grid(True)
plt.show()

best_k_index = np.argmax(cross_validation_scores)
best_k = k_list[best_k_index]
print(f"Best k from CV: {best_k}")

final_knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', p=3)
final_knn.fit(X_train_scaled, y_train)

y_pred = final_knn.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_pred)
print(f"Test-set Accuracy: {test_acc:.3f}")