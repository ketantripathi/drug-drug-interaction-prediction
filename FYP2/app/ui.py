import streamlit as st
import requests

st.set_page_config(page_title="Drug Interaction AI", layout="centered")

st.title("💊 Drug Interaction AI")

drug1 = st.text_input("Enter Drug A")
drug2 = st.text_input("Enter Drug B")

if st.button("Analyze"):

    if not drug1 or not drug2:
        st.warning("Please enter both drugs.")
    else:
        try:
            url = f"http://127.0.0.1:8000/predict?d1={drug1}&d2={drug2}"
            r = requests.get(url)

            # -------------------
            # RESPONSE CHECK
            # -------------------
            if r.status_code != 200:
                st.error(f"API Error: {r.status_code}")
                st.text(r.text)
                st.stop()

            result = r.json()

            if "error" in result:
                st.error(result["error"])
                st.stop()

            st.subheader("🔍 Prediction Result")

            # -------------------
            # MOLECULAR STRUCTURES
            # -------------------
            viz = result.get("visualization", {})

            st.subheader("🧬 Molecular Structures")

            col1, col2 = st.columns(2)

            if viz.get("drug1"):
                col1.image(
                    f"data:image/png;base64,{viz['drug1']}",
                    caption="Drug A Structure"
                )

            if viz.get("drug2"):
                col2.image(
                    f"data:image/png;base64,{viz['drug2']}",
                    caption="Drug B Structure"
                )

            # -------------------
            # INTERACTION RESULT
            # -------------------
            if result.get("prediction") == 1:
                st.error("⚠️ Interaction Likely")
            else:
                st.success("✅ No Strong Interaction")

            # -------------------
            # BASIC INFO
            # -------------------
            st.write(f"**Confidence:** {result.get('confidence', 0) * 100:.2f}%")
            st.write("**Interaction Type:**", result.get("interaction_type"))
            st.write("**Mechanism:**", result.get("mechanism"))

            # -------------------
            # REASONING
            # -------------------
            reasoning = result.get("reasoning_path", [])
            st.write("**Reasoning Path:**")
            st.write(" → ".join(reasoning) if reasoning else "No path found")

            # -------------------
            # RISK ANALYSIS
            # -------------------
            st.write("**Severity:**", result.get("severity"))
            st.write("**Harm Score:**", result.get("harm_score"))
            st.write("**Risk Level:**", result.get("risk_level"))

            st.write("**Side Effects:**")
            for e in result.get("side_effects", []):
                st.write("-", e)

            # -------------------
            # EXPLANATION
            # -------------------
            st.write("**Explanation:**")
            st.info(result.get("explanation"))

            # -------------------
            # NEW DRUG DISPLAY
            # -------------------
            new_drug = result.get("new_drug", {})

            if new_drug.get("generated"):

                st.subheader("🧪 New Drug Candidate")

                # SMILES
                st.write("**SMILES:**", new_drug.get("smiles"))

                # Structure
                if viz.get("new_drug"):
                    st.image(
                        f"data:image/png;base64,{viz['new_drug']}",
                        caption="Generated Drug Structure"
                    )

                # Analysis
                analysis = new_drug.get("analysis")

                if analysis:
                    st.write(
                        "**Predicted Interaction Score:**",
                        round(analysis.get("interaction_score", 0), 3)
                    )

        except Exception as e:
            st.error(f"Error: {str(e)}")