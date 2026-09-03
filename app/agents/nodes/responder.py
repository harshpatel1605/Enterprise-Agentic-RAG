import logfire
from app.agents.state import AgentState
from app.gateway import portkey_client, extract_cache_status


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """
    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "OFF_TOPIC":
        refusal = (
            "I'm sorry, but I can't help with that. I am a specialized Enterprise Assistant "
            "focused on Kubernetes, Intel hardware, and enterprise networking. "
            "You can ask me questions about architecture, deployment, pod autoscaling, or network configurations!"
        )
        return {
            "final_answer": refusal,
            "status": "Off-topic query politely declined.",
            "plan": state["plan"],
            "messages": [{"role": "assistant", "content": refusal}]
        }

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are a specialized Enterprise IT Assistant.
        Your area of expertise is STRICTLY limited to:
        1. Kubernetes (architecture, pods, scaling, HPA/VPA, operators, YAML manifests, cluster deployments)
        2. Intel Hardware (Xeon CPUs, FPGAs, SRIOV, DPDK, server accelerators, memory architecture)
        3. Enterprise Networking (BGP, SDN, VLANs, routing protocols, high-performance switches)

        CRITICAL INSTRUCTIONS:
        - If the user asks what topics you cover, what you can do, or what they can ask, state clearly and specifically that you specialize ONLY in Kubernetes, Intel hardware, and enterprise networking. Provide concrete examples of technical questions in those three areas.
        - NEVER claim or list topics outside this scope (do NOT mention ERP, CRM, HR, business strategy, finance, marketing, or general chatbots).
        - Be concise, professional, and helpful.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("✍️ LLM Synthesis"):
        try:
            response = portkey_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state["plan"] + ["Cache: Hit ⚡"]
                status = "Cache hit — instant response."
            else:
                logfire.info("✅ Response synthesised via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e
