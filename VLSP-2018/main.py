import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import re
import warnings
from underthesea import sent_tokenize
warnings.simplefilter(action='ignore', category=FutureWarning)
file = "vlsp2018_hotel/1-VLSP2018-SA-Hotel-train.csv"
file2 = "vlsp2018_hotel/2-VLSP2018-SA-Hotel-dev.csv"
file3 = "vlsp2018_restaurant/1-VLSP2018-SA-Restaurant-train.csv"

df = pd.read_csv(file)

def perform_eda(file_path, title):
    # 1. Đọc dữ liệu (Sử dụng utf-8)
    df = pd.read_csv(file_path, encoding='utf-8')

    # Tách cột Review và các cột nhãn (Aspect#Sentiment)
    review_col = 'Review'
    label_cols = [col for col in df.columns if col != review_col]

    # --- 1: ĐẾM SỐ MẪU MỖI LỚP ---
    sentiment_counts = df[label_cols].melt(var_name='Aspect', value_name='Sentiment')
    sentiment_summary = sentiment_counts[sentiment_counts['Sentiment'] != 0]['Sentiment'].value_counts().sort_index()
    
    label_map = {1: 'Positive', 2: 'Negative', 3: 'Neutral'}
    sentiment_summary.index = sentiment_summary.index.map(label_map)

    # --- 2: ĐỘ DÀI CÂU (Word count) ---
    df['word_count'] = df[review_col].apply(lambda x: len(str(x).split()))

    # --- 3: TỪ XUẤT HIỆN NHIỀU NHẤT ---
    all_words_raw = " ".join(df[review_col].astype(str)).lower()
    all_words_clean = re.sub(r'[^\w\s]', '', all_words_raw) # Xóa dấu câu
    words = all_words_clean.split()
    top_words = Counter(words).most_common(20)

    # --- TRỰC QUAN HÓA ---
    # Đổi thành 1 hàng 4 cột để thêm WordCloud
    fig, axes = plt.subplots(1, 4, figsize=(25, 8))

    # Plot 1: Phân bố nhãn cảm xúc
    sns.barplot(x=sentiment_summary.index, y=sentiment_summary.values, ax=axes[0], palette='viridis', hue=sentiment_summary.index, legend=False)
    axes[0].set_title(f'Phân bố nhãn cảm xúc ({title})')
    axes[0].set_ylabel('Số lượng mẫu')

    # Plot 2: Phân bố độ dài câu
    sns.histplot(df['word_count'], bins=30, kde=True, ax=axes[1], color='orange')
    axes[1].set_title(f'Phân bố độ dài câu ({title})')
    axes[1].set_xlabel('Số lượng từ')

    # Plot 3: Top 10 từ phổ biến
    top_df = pd.DataFrame(top_words[:10], columns=['Word', 'Count'])
    sns.barplot(x='Count', y='Word', data=top_df, ax=axes[2], palette='magma', hue='Word', legend=False)
    axes[2].set_title(f'Top 10 từ xuất hiện nhiều nhất ({title})')

    # Plot 4: WordCloud
    wc = WordCloud(width=800, height=800, background_color='white', colormap='plasma').generate(all_words_raw)
    axes[3].imshow(wc, interpolation='bilinear')
    axes[3].axis('off')
    axes[3].set_title(f'WordCloud ({title})')

    plt.tight_layout()
    plt.show()

    # In thông tin thống kê cơ bản
    print(f"Tổng số dòng: {len(df)}")
    print(f"Độ dài câu trung bình: {df['word_count'].mean():.2f} từ")
    print("-" * 30)

perform_eda(file3,"Restaurant Dataset")
