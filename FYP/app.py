# # app.py

# import gradio as gr
# from biobert_retrieval import retrieve_interaction

# def ui_predict(drug1, drug2):
#     drug1 = drug1.strip()
#     drug2 = drug2.strip()

#     if not drug1 or not drug2:
#         return "Please enter both drug names.", []

#     result = retrieve_interaction(drug1, drug2)

#     if result["status"] == "invalid_drug":
#         return result["message"], []

#     if result["status"] == "no_confident_match":
#         return result["message"], []

#     # Build a nice text summary + table of top results
#     summary = result["message"]

#     table = []
#     for r in result["results"]:
#         table.append([
#             r["drug1"],
#             r["drug2"],
#             f"{r['similarity']:.3f}",
#             r["description"]
#         ])

#     return summary, table

# with gr.Blocks(title="BioBERT-based Drug–Drug Interaction Explorer") as demo:
#     gr.Markdown(
#         """
#         # 💊 BioBERT-based Drug–Drug Interaction Explorer  
#         Enter two **valid real drugs** to see known or similar interaction descriptions.  
#         The system will **refuse unknown drug names** and will also abstain if it has **no confident match**.
#         """
#     )

#     with gr.Row():
#         drug1 = gr.Text(label="Drug 1")
#         drug2 = gr.Text(label="Drug 2")

#     submit_btn = gr.Button("Check Interaction")

#     output_msg = gr.Textbox(label="Status / Summary")
#     output_table = gr.Dataframe(
#         headers=["Drug 1", "Drug 2", "Similarity", "Known Interaction Description"],
#         datatype=["str", "str", "number", "str"],
#         row_count=(0, "dynamic"),
#         col_count=4
#     )

#     submit_btn.click(
#         fn=ui_predict,
#         inputs=[drug1, drug2],
#         outputs=[output_msg, output_table]
#     )

# demo.launch(server_name="0.0.0.0", server_port=7860)

# import gradio as gr
# import pandas as pd
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# import seaborn as sns
# from predict import predict_ddi


# # ========================================================
# #  LOAD DATA AND METRICS
# # ========================================================
# TEST_RESULTS_PATH = "test_results.csv"
# DATA_PATH = "data/db_drug_interactions.csv"
# EVAL_METRICS_PATH = "eval_metrics.json"   # Optional if you saved metrics


# df = pd.read_csv(DATA_PATH)
# df = df.dropna(subset=["Drug 1", "Drug 2", "Interaction Description"])

# test_df = None
# try:
#     test_df = pd.read_csv(TEST_RESULTS_PATH)
#     print("Loaded test_results.csv for dashboard.")
# except:
#     print("⚠ test_results.csv not found — evaluation tab will be limited.")


# # ========================================================
# #  FUNCTIONS FOR DASHBOARD
# # ========================================================

# # --- 1️⃣ Prediction Function ---
# def run_prediction(drug1, drug2):
#     result = predict_ddi(drug1, drug2)
#     return f"### 🔍 Predicted Interaction\n{result}"


# # --- 2️⃣ Evaluation Metrics Display ---
# def load_metrics():
#     if test_df is None:
#         return "Evaluation results not found."

#     # If you stored rouge metrics in JSON
#     try:
#         import json
#         metrics = json.load(open("eval_metrics.json", "r"))
#         return (
#             f"### 📊 Model Evaluation Metrics\n"
#             f"- **ROUGE-1:** {metrics['rouge1']:.4f}\n"
#             f"- **ROUGE-2:** {metrics['rouge2']:.4f}\n"
#             f"- **ROUGE-L:** {metrics['rougeL']:.4f}\n"
#         )
#     except:
#         return "ROUGE metrics file not found."


# # --- 3️⃣ Visualization: Top Drugs ---
# def plot_top_drugs():
#     plt.figure(figsize=(10, 5))
#     top_drugs = pd.concat([df['Drug 1'], df['Drug 2']]).value_counts().head(15)
#     sns.barplot(x=top_drugs.values, y=top_drugs.index)
#     plt.title("Top 15 Most Frequent Drugs")
#     plt.xlabel("Frequency")
#     plt.ylabel("Drug")

#     return plt.gcf()


# # --- 4️⃣ Visualization: Word Cloud ---
# def plot_wordcloud():
#     text = " ".join(df["Interaction Description"].values)
#     wordcloud = WordCloud(width=900, height=400).generate(text)

#     plt.figure(figsize=(12, 6))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis("off")
#     plt.title("Common Words in Interaction Descriptions")

#     return plt.gcf()


# # ========================================================
# #  BUILD DASHBOARD UI
# # ========================================================

# # --- Tab 1: Prediction ---
# predict_tab = gr.Interface(
#     fn=run_prediction,
#     inputs=[
#         gr.Text(label="Drug 1"),
#         gr.Text(label="Drug 2")
#     ],
#     outputs=gr.Markdown(),
#     title="DDI Prediction",
#     description=(
#         "Enter two drug names as they appear in the DrugBank dataset. "
#         "The system will first verify that the drugs and their pair exist "
#         "in the training data, and only then generate an interaction description."
#     )
# )

# # --- Tab 2: Evaluation Metrics ---
# metrics_tab = gr.Interface(
#     fn=load_metrics,
#     inputs=[],
#     outputs=gr.Markdown(),
#     title="Model Performance",
#     description="Evaluation metrics computed on the 20% held-out dataset."
# )

# # --- Tab 3: Visualization ---
# visual_tab = gr.Interface(
#     fn=lambda choice: plot_top_drugs() if choice == "Top Drugs" else plot_wordcloud(),
#     inputs=gr.Radio(["Top Drugs", "Word Cloud"], label="Visualization Type"),
#     outputs=gr.Plot(),
#     title="Dataset Visual Insights",
#     description="Explore dataset distribution and important patterns."
# )

# # --- Tab 4: Browse Full Test Results ---
# table_tab = gr.Interface(
#     fn=lambda: test_df if test_df is not None else pd.DataFrame({"Error": ["test_results.csv missing"]}),
#     inputs=None,
#     outputs=gr.DataFrame(),
#     title="Test Dataset Results",
#     description="Browse actual vs predicted interactions."
# )

# # ========================================================
# #  COMBINE ALL TABS IN ONE DASHBOARD
# # ========================================================

# demo = gr.TabbedInterface(
#     interface_list=[predict_tab, metrics_tab, visual_tab, table_tab],
#     tab_names=["🔮 Predict Interaction", "📊 Evaluation Metrics", "📈 Visual Insights", "📄 Test Results"]
# )

# demo.launch(server_name="0.0.0.0", server_port=7860)

import gradio as gr
from predict import predict_ddi_hybrid

def ui_predict(drug1, drug2):
    result = predict_ddi_hybrid(drug1, drug2)
    return (
        result["status"],
        result["kb_interaction"],
        result["model_prediction"],
        result["similar_examples"],
        result["notes"],
    )

with gr.Blocks(title="Hybrid Drug–Drug Interaction Explorer") as demo:
    gr.Markdown(
        """
        # Hybrid Drug–Drug Interaction Explorer
        """
    )

    with gr.Row():
        drug1_in = gr.Text(label="Drug 1")
        drug2_in = gr.Text(label="Drug 2")

    run_btn = gr.Button("Check Interaction")

    status_out = gr.Textbox(label="Status", lines=2)
    kb_out = gr.Textbox(
        label="Known interaction from dataset (if available)",
        lines=6,
    )
    pred_out = gr.Textbox(
        label="Model-based prediction (for valid, unseen pairs)",
        lines=6,
    )
    sims_out = gr.Textbox(
        label="Similar known interactions used as evidence",
        lines=10,
    )
    notes_out = gr.Textbox(
        label="Notes / Warnings",
        lines=4,
    )

    run_btn.click(
        fn=ui_predict,
        inputs=[drug1_in, drug2_in],
        outputs=[status_out, kb_out, pred_out, sims_out, notes_out],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
