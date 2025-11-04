import streamlit as st
from langchain_helper import get_gemini_sql_agent  # your helper

st.set_page_config(page_title="SagarBhat T-Shirts: SQL Q&A", page_icon="👕")
st.title("SagarBhat T-Shirts: Database Q&A 👕")

question = st.text_input("Enter your question:")

if question:
    with st.spinner("Querying database... ⏳"):
        agent = get_gemini_sql_agent()

        try:
            # For standard LangChain agent
            response = agent.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )

            for step in agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                stream_mode="values",
            ):
                step["messages"][-1].pretty_print()

            # Extract the answer text
            if isinstance(response, dict) and "messages" in response:
                final_answer = response["messages"][-1].content
            elif isinstance(response, list) and len(response) > 0:
                final_answer = response[-1].get("text", "")
                
            else:
                final_answer = str(response)

            st.success("✅ Done!")
            st.subheader("Answer:")
            st.write(final_answer)

        except Exception as e:
            st.error(f"❌ Error: {e}")
