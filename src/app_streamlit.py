import traceback

import streamlit as st

from .rag_pipeline import RAGPipeline


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    """
    Создаём RAG-пайплайн один раз и переиспользуем
    между запросами (экономит время и токены).
    """
    return RAGPipeline(top_k=4)


def main():
    st.set_page_config(page_title="IT Support RAG Assistant", page_icon="💻")
    st.title("💻 IT Support RAG Assistant")
    st.write(
        "Ask any IT support question related to Wi-Fi, VPN, email, printers or password policy. "
        "The assistant will search in the internal knowledge base and answer using that context."
    )

    question = st.text_area(
        "Your question:",
        placeholder="Example: How can I connect to corporate Wi-Fi on Windows?",
        height=100,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        temperature = st.slider(
            "Creativity (temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
        )
    with col2:
        top_k = st.slider(
            "Number of context chunks (top_k)",
            min_value=1,
            max_value=8,
            value=4,
            step=1,
        )

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        try:
            with st.spinner("Thinking..."):
                pipeline = get_pipeline()
                # при необходимости обновим top_k на лету
                pipeline.top_k = top_k

                result = pipeline.answer_question(question)
                answer = result["answer"]
                docs = result["documents"]

            st.subheader("Answer")
            st.markdown(answer)

            st.subheader("Retrieved context")
            if not docs:
                st.info("No relevant documents were found in the knowledge base.")
            else:
                for i, doc in enumerate(docs, start=1):
                    meta = doc.get("metadata", {}) or {}
                    score = doc.get("score", 0.0)

                    title = meta.get("title", "(no title)")
                    source_type = meta.get("source_type", "unknown")
                    category = meta.get("category", "unknown")

                    with st.expander(
                        f"[Doc {i}] {title} "
                        f"(source={source_type}, category={category}, score={score:.3f})"
                    ):
                        st.write(doc.get("text", ""))

        except Exception as e:
            st.error("Error while processing the request.")
            st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))


if __name__ == "__main__":
    main()
