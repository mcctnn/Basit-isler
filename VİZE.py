# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:23:43 2024

@author: cetin
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from boruta import BorutaPy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# datayı yükleme
file_path = 'C:/Users/cetin/Desktop/excelData1.xlsx'
excel_data = pd.ExcelFile(file_path)

print("Sheet Names:", excel_data.sheet_names)

# her dataframe için sheet yükle
yuklu_df = pd.read_excel(file_path, sheet_name='yuklu')
yuksuz_df = pd.read_excel(file_path, sheet_name='yuksuz')
akuSarj_df = pd.read_excel(file_path, sheet_name='akuSarj')

#  unique sütun adı oluşturma
def make_unique(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

# unique kelime mi
yuklu_df = make_unique(yuklu_df)
yuksuz_df = make_unique(yuksuz_df)
akuSarj_df = make_unique(akuSarj_df)

# aynı isimlileri değiştirme
yuklu_df = yuklu_df.loc[:, ~yuklu_df.columns.duplicated()]
yuksuz_df = yuksuz_df.loc[:, ~yuksuz_df.columns.duplicated()]
akuSarj_df = akuSarj_df.loc[:, ~akuSarj_df.columns.duplicated()]

# verilen dataframe için ort,medyan min max hesaplama
def calculate_statistics(df):
    statistics = df.describe().T[['mean', '50%', 'min', 'max']].rename(columns={'50%': 'median'})
    return statistics

# her dataframe için istatistik hesaplama
yuklu_stats = calculate_statistics(yuklu_df)
yuksuz_stats = calculate_statistics(yuksuz_df)
akuSarj_stats = calculate_statistics(akuSarj_df)

# istatistikleri yaz
print("Yüklü Durumu İstatistikleri:\n", yuklu_stats)
print("Yüksüz Durumu İstatistikleri:\n", yuksuz_stats)
print("Akü Şarj Durumu İstatistikleri:\n", akuSarj_stats)

# istatistik için kullanıcak olan grafikler oluşturan fonksiyon
def plot_statistics(stats_df, title):
    stats_df.plot(kind='bar', figsize=(10, 6))
    plt.title(title)
    plt.ylabel('Değerler')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# her dataframe için plot oluşturma
plot_statistics(yuklu_stats, 'Yüklü Durumu İstatistikleri')
plot_statistics(yuksuz_stats, 'Yüksüz Durumu İstatistikleri')
plot_statistics(akuSarj_stats, 'Akü Şarj Durumu İstatistikleri')

def plot_correlation(df, title):
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title(f'{title} Korelasyon Matrisi')
    plt.show()

# Yüklü, Yüksüz ve Akü Şarj sayfaları için korelasyon analizi
plot_correlation(yuklu_df, 'Yüklü Durumu')
plot_correlation(yuksuz_df, 'Yüksüz Durumu')
plot_correlation(akuSarj_df, 'Akü Şarj Durumu')

def t_test(yuksuz, yuklu, param):
    t_stat, p_value = stats.ttest_ind(yuksuz[param], yuklu[param], equal_var=False)
    print(f'T-Test for {param}: t-stat={t_stat}, p-value={p_value}')
    if p_value < 0.05:
        print(f"{param} için iki grup arasında anlamlı bir fark vardır (p < 0.05).")
    else:
        print(f"{param} için iki grup arasında anlamlı bir fark yoktur (p >= 0.05).")

# Yüksüz ve Yüklü koşullar için T-Testi
for column in yuklu_df.columns:
    if column in yuksuz_df.columns:
        print(f"Testing column: {column}")  # Hangi sütunun test edildiğini göster
        t_test(yuksuz_df, yuklu_df, column)
    else:
        print(f"{column} sütunu yuksuz_df'de bulunmamaktadır.")  # Sütunun bulunmadığını göster

def visualize_t_test(yuksuz, yuklu, param):
    combined_df = pd.DataFrame({
        'Değerler': pd.concat([yuksuz[param], yuklu[param]]),
        'Grup': ['Yüksüz'] * len(yuksuz[param]) + ['Yüklü'] * len(yuklu[param])
    })
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Grup', y='Değerler', data=combined_df)
    plt.title(f'{param} için Yüksüz ve Yüklü Durumu Karşılaştırması')
    plt.show()

# Her bir parametre için kutu grafiği oluşturma
for column in yuklu_df.columns:
    if column in yuksuz_df.columns:
        visualize_t_test(yuksuz_df, yuklu_df, column)

# 3. ANOVA (Üç grup için - Yüksüz, Yüklü, Akü Şarj)
def anova_test(yuksuz, yuklu, akuSarj, param):
    f_stat, p_value = stats.f_oneway(yuksuz[param], yuklu[param], akuSarj[param])
    print(f'ANOVA for {param}: f-stat={f_stat}, p-value={p_value}')
    if p_value < 0.05:
        print(f"{param} için üç grup arasında anlamlı bir fark vardır (p < 0.05).")
    else:
        print(f"{param} için üç grup arasında anlamlı bir fark yoktur (p >= 0.05).")

# Üç koşul için ANOVA
for column in yuklu_df.columns:
    if column in yuksuz_df.columns and column in akuSarj_df.columns:
        anova_test(yuksuz_df, yuklu_df, akuSarj_df, column)

# ANOVA görselleştirme (çubuk grafikleri)
def visualize_anova(yuksuz, yuklu, akuSarj, param):
    means = [yuksuz[param].mean(), yuklu[param].mean(), akuSarj[param].mean()]
    errors = [yuksuz[param].std(), yuklu[param].std(), akuSarj[param].std()]
    
    plt.figure(figsize=(8, 6))
    plt.bar(['Yüksüz', 'Yüklü', 'Akü Şarj'], means, yerr=errors, capsize=5, color=['blue', 'orange', 'green'])
    plt.title(f'{param} için Yüksüz, Yüklü ve Akü Şarj Durumu Karşılaştırması')
    plt.ylabel('Ortalama Değer')
    plt.tight_layout()
    plt.show()

# Her bir parametre için çubuk grafikleri oluşturma
for column in yuklu_df.columns:
    if column in yuksuz_df.columns and column in akuSarj_df.columns:
        visualize_anova(yuksuz_df, yuklu_df, akuSarj_df, column)
        plt.show()

# Modelleme bölümü
X = yuklu_df.drop(['Alternatör Akımı', 'Alternatör Gerilimi', 'Sıcaklık'], axis=1)  # Giriş değişkenleri
y_akim = yuklu_df['Alternatör Akımı']  # Çıktı değişkeni olarak alternatör akımı
y_gerilim = yuklu_df['Alternatör Gerilimi']  # Alternatör gerilimi
y_sicaklik = yuklu_df['Sıcaklık']  # Sıcaklık

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train_akim, y_test_akim = train_test_split(X, y_akim, test_size=0.2, random_state=42)

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train_akim)

# Özelliklerin önemi (MDI - Mean Decrease Impurity)
importances = rf.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print("Özelliklerin Önemi:")
print(feature_importances.sort_values(by='Importance', ascending=False))

# Karar Ağacı ile Tahmin
dt_reg = DecisionTreeRegressor(random_state=42)
dt_reg.fit(X_train, y_train_akim)

# Tahmin edilen değerler
y_pred_akim = dt_reg.predict(X_test)
print("Tahmin edilen alternatör akımı:", y_pred_akim)

# Sınıflandırma için veri hazırlama (alternatör akımı medyanına göre)
y_class = (y_akim >= y_akim.median()).astype(int)  # Sınıflar oluşturma (0 ve 1)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

dt_class = DecisionTreeClassifier(random_state=42)
dt_class.fit(X_train_class, y_train_class)

# Tahmin edilen sınıflar
y_pred_class = dt_class.predict(X_test_class)
print("Tahmin edilen sınıflar:", y_pred_class)

# Boruta Algoritması
boruta_selector = BorutaPy(
    estimator=rf,
    n_estimators='auto',
    verbose=2,
    random_state=42
)
boruta_selector.fit(X.values, y_class.values)

# Seçilen değişkenleri yazdırma
print("Seçilen değişkenler:", X.columns[boruta_selector.support_])

# K-Means Kümeleme
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# PCA ile Görselleştirme
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('K-Means Kümeleme ile PCA Görselleştirme')
plt.xlabel('PCA Bileşeni 1')
plt.ylabel('PCA Bileşeni 2')
plt.colorbar(label='Küme Etiketleri')
plt.show()
print(akuSarj_df.columns)