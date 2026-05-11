import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ==================================================
# MAKE DIRECTORY TO SAVE PLOTS
# ==================================================
os.makedirs("plots", exist_ok=True)

# ==================================================
# 1️⃣ LOAD TEST RESULTS
# ==================================================
TEST_RESULTS = "test_results.csv"
DATA_PATH = "data/db_drug_interactions.csv"

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["Drug 1", "Drug 2", "Interaction Description"])

test_df = pd.read_csv(TEST_RESULTS)
print("Loaded test dataset:", len(test_df))

# ==================================================
# 2️⃣ DRUG FREQUENCY ANALYSIS
# ==================================================
plt.figure(figsize=(10,5))
top_drugs = pd.concat([df['Drug 1'], df['Drug 2']]).value_counts().head(20)
sns.barplot(x=top_drugs.values, y=top_drugs.index, palette='viridis')
plt.title("Top 20 Most Frequent Drugs")
plt.xlabel("Frequency")
plt.ylabel("Drug Name")
plt.tight_layout()
plt.savefig("plots/top_drugs.png")
plt.close()


# ==================================================
# 3️⃣ WORDCLOUD OF ACTUAL INTERACTIONS
# ==================================================
text_actual = " ".join(df["Interaction Description"].values)
wordcloud = WordCloud(width=900, height=400, background_color='white').generate(text_actual)

plt.figure(figsize=(12,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud - Actual DDI Descriptions")
plt.savefig("plots/wordcloud_actual.png")
plt.close()


# ==================================================
# 4️⃣ WORDCLOUD OF PREDICTED INTERACTIONS
# ==================================================
text_pred = " ".join(test_df["Predicted Interaction"].values)
wordcloud_pred = WordCloud(width=900, height=400, background_color='white').generate(text_pred)

plt.figure(figsize=(12,5))
plt.imshow(wordcloud_pred, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud - Predicted DDI Descriptions")
plt.savefig("plots/wordcloud_predicted.png")
plt.close()


# ==================================================
# 5️⃣ COSINE SIMILARITY ANALYSIS
# ==================================================
vectorizer = TfidfVectorizer()
pairs = vectorizer.fit_transform(
    test_df["Interaction Description"].tolist() + test_df["Predicted Interaction"].tolist()
)

actual_vectors = pairs[:len(test_df)]
pred_vectors = pairs[len(test_df):]

similarities = cosine_similarity(actual_vectors, pred_vectors).diagonal()
test_df["Similarity Score"] = similarities


plt.figure(figsize=(10,5))
sns.histplot(similarities, bins=30, kde=True, color='purple')
plt.title("Distribution of Cosine Similarity (Actual vs Predicted)")
plt.xlabel("Similarity Score (0 to 1)")
plt.ylabel("Frequency")
plt.savefig("plots/similarity_distribution.png")
plt.close()


# ==================================================
# 6️⃣ BEST & WORST SAMPLES
# ==================================================
best_samples = test_df.nlargest(5, "Similarity Score")
worst_samples = test_df.nsmallest(5, "Similarity Score")

best_samples.to_csv("best_predictions.csv", index=False)
worst_samples.to_csv("worst_predictions.csv", index=False)


# ==================================================
# 7️⃣ LENGTH COMPARISON
# ==================================================
test_df["Actual Length"] = test_df["Interaction Description"].apply(lambda x: len(str(x).split()))
test_df["Predicted Length"] = test_df["Predicted Interaction"].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(10,5))
sns.kdeplot(test_df["Actual Length"], label="Actual", fill=True, color="green")
sns.kdeplot(test_df["Predicted Length"], label="Predicted", fill=True, color="red")
plt.title("Length Comparison: Actual vs Predicted Descriptions")
plt.xlabel("Word Count")
plt.legend()
plt.savefig("plots/length_comparison.png")
plt.close()


# ==================================================
# 8️⃣ SIMILARITY VS SAMPLE INDEX
# ==================================================
plt.figure(figsize=(10,5))
plt.scatter(range(len(test_df)), test_df["Similarity Score"], s=12, alpha=0.7)
plt.title("Similarity Score per DDI Sample")
plt.xlabel("Sample Index")
plt.ylabel("Similarity Score")
plt.savefig("plots/similarity_scatter.png")
plt.close()


# ==================================================
# 9️⃣ *** NEW IMPORTANT GRAPH ***
#     SIMILARITY VS ACTUAL LENGTH HEATMAP
# ==================================================
plt.figure(figsize=(8,6))
sns.histplot(
    x=test_df["Actual Length"],
    y=test_df["Similarity Score"],
    bins=20,
    pmax=0.9,
    cmap="viridis"
)
plt.title("Heatmap: Actual Text Length vs Similarity Score")
plt.savefig("plots/heatmap_length_similarity.png")
plt.close()


# ==================================================
# 🔟 *** NEW GRAPH ***
#     CORRELATION MATRIX
# ==================================================
corr = test_df[["Similarity Score", "Actual Length", "Predicted Length"]].corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("plots/correlation_matrix.png")
plt.close()


# ==================================================
# 1️⃣1️⃣ *** NEW GRAPH ***
#      ERROR DISTRIBUTION (1 - similarity)
# ==================================================
errors = 1 - test_df["Similarity Score"]

plt.figure(figsize=(10,5))
sns.histplot(errors, bins=30, kde=True, color="red")
plt.title("Error Distribution (1 - Similarity Score)")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.savefig("plots/error_distribution.png")
plt.close()


# ==================================================
# 1️⃣2️⃣ SAVE FINAL EXTENDED DATA
# ==================================================
test_df.to_csv("test_results_with_similarity.csv", index=False)
print("\n✅ Extended analysis saved to test_results_with_similarity.csv")
print("📁 All plots saved in /plots folder")
