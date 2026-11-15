
import streamlit as st

from rag import answer


# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AsterHome AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AsterHome AI Assistant")

st.caption(
    "Your intelligent assistant for AsterHome products and support."
)

st.caption(
    "Demo project | Products, prices, and policies are fictional."
)

st.divider()


# --------------------------------------------------
# Helper: Display Sources Safely
# --------------------------------------------------

def display_sources(sources):

    for index, source in enumerate(sources, start=1):

        if not source:
            continue

        if isinstance(source, dict):

            file_name = source.get("source") or source.get("file_name")
            page = source.get("page")

            if file_name:
                st.markdown(
                    f"**Source {index}: {file_name}**"
                )

            if page is not None:
                st.caption(f"Page: {page}")

            if not file_name:
                st.write(source)

        elif isinstance(source, (tuple, list)):

            if len(source) != 2:
                continue

            file_name, page = source

            if not file_name:
                continue

            st.markdown(
                f"**Source {index}: {file_name}**"
            )

            st.caption(f"Page: {page}")

        else:

            st.markdown(
                f"**Source {index}:** {source}"
            )


# --------------------------------------------------
# 2. Initialize Session State
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""

if "history" not in st.session_state:
    st.session_state.history = []


# --------------------------------------------------
# 3. Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Settings")

    if st.button("🗑️ Clear Conversation", use_container_width=True):

        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.conversation_summary = ""

        st.rerun()

    st.divider()

    st.subheader("Conversation Memory")

    st.caption(
        "The assistant uses recent conversation history "
        "and a summary of older messages."
    )

    with st.expander("View Conversation Summary"):

        if st.session_state.conversation_summary:
            st.write(st.session_state.conversation_summary)
        else:
            st.write("No summary available yet.")

    with st.expander("View Recent History"):

        st.write(st.session_state.history)


# --------------------------------------------------
# 4. Welcome Message
# --------------------------------------------------

if not st.session_state.messages:

    st.info(
        "👋 Welcome to AsterHome!\n\n"
        "You can ask about products, prices, specifications, "
        "installation, and support policies."
    )


# --------------------------------------------------
# 5. Display Previous Messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display sources
        if message["role"] == "assistant" and message.get("sources"):

            with st.expander("📚 View Sources"):

                display_sources(message["sources"])

        # Display retrieved contexts
        if message["role"] == "assistant" and message.get("contexts"):

            with st.expander("🔍 Retrieved Chunks"):

                for index, context in enumerate(
                    message["contexts"], start=1
                ):

                    st.markdown(f"**Chunk {index}**")

                    st.write(context)

                    st.divider()

        # Display RAG metadata
        if message["role"] == "assistant" and message.get("route"):

            with st.expander("⚙️ RAG Information"):

                st.write(f"**Route:** {message['route']}")

                st.write(
                    f"**Hop Type:** {message.get('hop_type')}"
                )

                st.write(
                    f"**Retried:** {message.get('retried')}"
                )

                if message.get("meta"):

                    meta = message["meta"]

                    st.write(f"**Model:** {meta.get('model')}")

                    st.write(
                        f"**Latency:** {meta.get('latency')} seconds"
                    )

                    st.write(
                        f"**Prompt Tokens:** {meta.get('prompt_tokens')}"
                    )

                    st.write(
                        f"**Completion Tokens:** "
                        f"{meta.get('completion_tokens')}"
                    )


# --------------------------------------------------
# 6. Chat Input
# --------------------------------------------------

question = st.chat_input(
    "Ask anything about AsterHome..."
)


# --------------------------------------------------
# 7. Process User Question
# --------------------------------------------------

if question:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Display user message
    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------
    # 8. Generate Answer Using Adaptive RAG
    # --------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                result = answer(
                    question,
                    history=st.session_state.history,
                    conversation_summary=(
                        st.session_state.conversation_summary
                    )
                )

                response = result["answer"]

                # Display answer
                st.markdown(response)


                # --------------------------------------------------
                # 9. Display Sources
                # --------------------------------------------------

                if result.get("sources"):

                    with st.expander("📚 View Sources"):

                        display_sources(result["sources"])


                # --------------------------------------------------
                # 10. Display Retrieved Chunks
                # --------------------------------------------------

                if result.get("contexts"):

                    with st.expander("🔍 Retrieved Chunks"):

                        for index, context in enumerate(
                            result["contexts"], start=1
                        ):

                            st.markdown(f"**Chunk {index}**")

                            st.write(context)

                            st.divider()


                # --------------------------------------------------
                # 11. Display Adaptive RAG Information
                # --------------------------------------------------

                with st.expander("⚙️ RAG Information"):

                    st.write(f"**Route:** {result.get('route')}")

                    st.write(
                        f"**Hop Type:** {result.get('hop_type')}"
                    )

                    st.write(
                        f"**Retry:** {result.get('retried')}"
                    )

                    if result.get("meta"):

                        meta = result["meta"]

                        st.write(f"**Model:** {meta.get('model')}")

                        st.write(
                            f"**Latency:** {meta.get('latency')} seconds"
                        )

                        st.write(
                            f"**Prompt Tokens:** {meta.get('prompt_tokens')}"
                        )

                        st.write(
                            f"**Completion Tokens:** "
                            f"{meta.get('completion_tokens')}"
                        )


                # --------------------------------------------------
                # 12. Save Conversation History
                # --------------------------------------------------

                st.session_state.history.append({
                    "role": "user",
                    "content": question
                })

                st.session_state.history.append({
                    "role": "assistant",
                    "content": response
                })


                # --------------------------------------------------
                # 13. Update Conversation Summary
                # --------------------------------------------------

                if len(st.session_state.history) > 6:

                    from rag import update_conversation_summary

                    old_messages = (
                        st.session_state.history[:-6]
                    )

                    try:

                        new_summary = update_conversation_summary(
                            st.session_state.conversation_summary,
                            old_messages
                        )

                        if new_summary:

                            st.session_state.conversation_summary = (
                                new_summary
                            )

                            st.session_state.history = (
                                st.session_state.history[-6:]
                            )

                    except Exception as error:

                        st.warning(
                            f"Summary update error: {error}"
                        )


            except Exception as error:

                response = (
                    "Sorry, something went wrong. "
                    "Please try again."
                )

                st.error(response)

                st.caption(f"Technical details: {error}")

                result = {}


    # --------------------------------------------------
    # 14. Save Assistant Message
    # --------------------------------------------------

    st.session_state.messages.append({

        "role": "assistant",

        "content": response,

        "sources": result.get("sources", []),

        "contexts": result.get("contexts", []),

        "route": result.get("route"),

        "hop_type": result.get("hop_type"),

        "retried": result.get("retried"),

        "meta": result.get("meta")

    })
