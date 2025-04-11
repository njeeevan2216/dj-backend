
# !pip install langchain langgraph langchain-community langchain-google-genai langchain-neo4j neo4j google-genai


# --- START OF MODIFIED CODE ---
import time
import re
import logging
from typing import Literal, List, Dict, Annotated, Any, Optional
from operator import add
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# LangChain Core imports
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.runnables import Runnable, RunnableSequence # Added RunnableSequence
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage # Added SystemMessage, AIMessage
from langchain_core.documents import Document # Added for RRF
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

# LangChain Community/Integration imports
from langchain_community.graphs import Neo4jGraph
from langchain_neo4j import Neo4jVector # For KG example selector
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema # For KG corrector
from neo4j.exceptions import CypherSyntaxError # For KG validation

# LLM Initialization (Assuming Google Generative AI)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from google import genai
    from google.genai import types
    # from dotenv import load_dotenv
    import os


    # Load environment variables from .env.local
    # load_dotenv(".env.local")
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    # AIzaSyAYSZyAaYQwXfE5OM9vf9yqlc7DV2AHjyQ
    import getpass

    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass(os.getenv('GEMINI_API_KEY'))

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", google_api_key=os.getenv('GEMINI_API_KEY'), temperature=0.7)
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=os.getenv('GEMINI_API_KEY'), model="models/embedding-001")

    print("Using ChatGoogleGenerativeAI and GoogleGenerativeAIEmbeddings.")
except Exception as e:
    print(f"Failed to initialize Google Generative AI components: {e}")
    llm = None
    embeddings = None

# Other necessary imports
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# LangGraph imports
from langgraph.graph import StateGraph, END

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
parser = StrOutputParser() # Global parser for some LLM outputs

# ==============================================================================
# == RAG Neo4j Setup (Specific Credentials for RAG)
# ==============================================================================
try:
    # Credentials for the RAG Neo4j Instance
    RAG_NEO4J_URL = "neo4j+s://17a2ec84.databases.neo4j.io" # Replace if different
    RAG_NEO4J_USER = "neo4j" # Replace if different
    # Use a secure way to handle passwords (env variable, secret manager)
    # RAG_NEO4J_PASSWORD = userdata.get('RAG_NEO4J_PASSWORD') # Example for Colab
    RAG_NEO4J_PASSWORD = "X1zgwwz3gxijzbOb6hAi6QtlqVnZPu6uITXWsG_qGfk" # Replace with actual password or secure method

    # Initialize the Neo4jGraph object specifically for RAG
    rag_neo4j_graph = Neo4jGraph(url=RAG_NEO4J_URL, username=RAG_NEO4J_USER, password=RAG_NEO4J_PASSWORD)
    # Verify connection
    rag_neo4j_graph.query("RETURN 1")
    print("RAG Neo4j connection successful.")
except Exception as e:
    logging.exception("Failed to connect to RAG Neo4j. RAG nodes requiring Neo4j will fail.")
    print(f"RAG NEO4J CONNECTION FAILED: {e}")
    rag_neo4j_graph = None

# ==============================================================================
# == Knowledge Graph (KG) Neo4j Setup (Specific Credentials for KG)
# ==============================================================================
try:
    # Credentials for the KG Neo4j Instance
    KG_NEO4J_URL = "neo4j+s://624367d3.databases.neo4j.io" # From KG code
    KG_NEO4J_USER = "neo4j" # From KG code
    # Use a secure way to handle passwords
    # KG_NEO4J_PASSWORD = userdata.get('KG_NEO4J_PASSWORD') # Example for Colab
    KG_NEO4J_PASSWORD = "Q149B4DKJTFyb_ASNh_CnSlbMIeUgb6XXuQLj4P2c5w" # From KG code

    # Initialize the Neo4jGraph object specifically for KG (with enhanced schema)
    kg_neo4j_graph = Neo4jGraph(
        url=KG_NEO4J_URL,
        username=KG_NEO4J_USER,
        password=KG_NEO4J_PASSWORD,
        enhanced_schema=True # Use enhanced schema for KG
    )
    # Verify connection
    kg_neo4j_graph.query("RETURN 1")
    print("KG Neo4j connection successful.")

    # --- KG Components Initialization (Requires KG Neo4j connection and LLM) ---
    if llm and embeddings and kg_neo4j_graph:
        # Few-shot examples for Cypher generation
        kg_examples = [
            {"question": "How many hospitals are there in Karnataka?", "query": "MATCH (h:Hospital)-[:LOCATED_IN]->(d:District)-[:PART_OF]->(s:State {state_name: 'KARNATAKA'}) RETURN count(h)"},
            {"question": "Which hospitals in Uttar Pradesh offer neurosurgery?", "query": "MATCH (h:Hospital)-[:OFFERS]->(s:Surgery)-[:BELONGS_TO]->(sc:SubCategory)-[:PART_OF]->(c:Category {category_name: 'Neurosurgery'}), (h)-[:LOCATED_IN]->(d:District)-[:PART_OF]->(s_state:State {state_name: 'UTTAR PRADESH'}) RETURN h.hospital_name"}, # Added state link for hospital
            {"question": "What is the cost of Craniotomy And Evacuation Of Hematoma Subdural?", "query": "MATCH (s:Surgery {surgery_name: 'Craniotomy And Evacuation Of Hematoma Subdural'})<-[o:OFFERS]-(h:Hospital) RETURN h.hospital_name, o.cost"}, # Return hospital name too
            {"question": "Which hospitals in Tamil Nadu are empaneled under PMJAY?", "query": "MATCH (h:Hospital {empanelment_type: 'PMJAY'})-[:LOCATED_IN]->(d:District)-[:PART_OF]->(s:State {state_name: 'TAMIL NADU'}) RETURN h.hospital_name"},
            {"question": "What specialties are available in Bihar?", "query": "MATCH (h:Hospital)-[:HAS_SPECIALTY]->(sp:Specialty), (h)-[:LOCATED_IN]->(d:District)-[:PART_OF]->(s:State {state_name: 'BIHAR'}) RETURN DISTINCT sp.specialty_name"},
            {"question": "How many private hospitals are there in Gujarat?", "query": "MATCH (h:Hospital)-[:HAS_TYPE]->(t:HospitalType) WHERE t.type_name CONTAINS 'Private' AND (h)-[:LOCATED_IN]->(:District)-[:PART_OF]->(:State {state_name: 'GUJARAT'}) RETURN count(h)"},
            {"question": "Which hospitals offer brain surgeries?", "query": "MATCH (h:Hospital)-[:OFFERS]->(s:Surgery)-[:BELONGS_TO]->(sc:SubCategory {sub_category_name: 'Brain'}) RETURN h.hospital_name"},
            {"question": "What is the empanelment status of Govt.DDWO, Sivagangai TN?", "query": "MATCH (h:Hospital {hospital_name: 'Govt.DDWO,Sivagangai TN.'}) RETURN h.application_status"},
            {"question": "Which surgeries are available in Jharkhand?", "query": "MATCH (h:Hospital)-[:OFFERS]->(s:Surgery), (h)-[:LOCATED_IN]->(d:District)-[:PART_OF]->(st:State {state_name: 'JHARKHAND'}) RETURN DISTINCT s.surgery_name"}, # Corrected state variable
            {"question": "How many hospitals have been onboarded for convergence schemes?", "query": "MATCH (h:Hospital {onboarded_for_convergence_scheme: 'Yes'}) RETURN count(h)"},
        ]

        
        kg_example_selector = SemanticSimilarityExampleSelector.from_examples(
            kg_examples, embeddings, Neo4jVector, k=3, input_keys=["question"],
            url=KG_NEO4J_URL, username=KG_NEO4J_USER, password=KG_NEO4J_PASSWORD,

        )
        print("KG Example Selector initialized.")

        # KG Text-to-Cypher Chain
        kg_text2cypher_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given an input question, convert it to a Cypher query. No pre-amble. Do not wrap the response in backticks. Respond with a Cypher statement only! Use the provided schema."),
            ("human", """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.
              Use ONLY the nodes, relationships and properties mentioned in the schema. Do NOT use anything not in the schema.
              Do not wrap the response in backticks. Respond with a Cypher statement only!

              Schema:
              {schema}

              Below are examples of questions and their Cypher queries:
              {fewshot_examples}

              User input: {question}
              Cypher query:""")
        ])
        kg_text2cypher_chain = kg_text2cypher_prompt | llm | StrOutputParser()
        print("KG Text2Cypher chain initialized.")

        # KG Cypher Validation Chain (Structured Output)
        kg_validate_cypher_system = "You are a Cypher expert reviewing a statement written by a potentially incorrect AI. Check for syntax, schema adherence, and question relevance."
        kg_validate_cypher_user = """Check the following for the Cypher statement:
        * Syntax errors?
        * Missing/undefined variables?
        * Node labels missing from the schema?
        * Relationship types missing from the schema?
        * Properties not included in the schema for the given node/relationship?
        * Does the query logically answer the question based on the schema?

        Schema:
        {schema}

        Question:
        {question}

        Cypher statement:
        {cypher}

        Return ONLY the JSON structure matching ValidateCypherOutput. Provide explanations for errors found.
        """
        kg_validate_cypher_prompt = ChatPromptTemplate.from_messages([
            ("system", kg_validate_cypher_system),
            ("human", kg_validate_cypher_user),
        ])
        class Property(BaseModel):
            node_label: str = Field(description="Node label")
            property_key: str = Field(description="Property key")
            property_value: str = Field(description="Property value used in the query")
        class ValidateCypherOutput(BaseModel):
            errors: Optional[List[str]] = Field(description="Syntax or semantic errors found, with explanations.")
            filters: Optional[List[Property]] = Field(description="Property filters applied in the query.")
        kg_validate_cypher_chain = kg_validate_cypher_prompt | llm.with_structured_output(ValidateCypherOutput)
        print("KG ValidateCypher chain initialized.")

        # KG Cypher Correction Chain
        # Use Schema objects for the corrector
        kg_corrector_schema = [Schema(el["start"], el["type"], el["end"]) for el in kg_neo4j_graph.structured_schema.get("relationships", [])]
        kg_cypher_query_corrector = CypherQueryCorrector(kg_corrector_schema)
        kg_correct_cypher_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Cypher expert correcting a statement based on provided errors and schema. Respond ONLY with the corrected Cypher query, no backticks or preamble."),
            ("human", """Correct the Cypher statement based on the errors and schema. Respond ONLY with the corrected Cypher query.

            Schema:
            {schema}

            Question:
            {question}

            Original Cypher:
            {cypher}

            Errors:
            {errors}

            Corrected Cypher query:""")
        ])
        kg_correct_cypher_chain = kg_correct_cypher_prompt | llm | StrOutputParser()
        print("KG CorrectCypher chain initialized.")

        # KG Final Answer Generation Chain
        kg_generate_final_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant interpreting database results."),
            ("human", """Use the following results retrieved from the database to provide a succinct answer to the user's original sub-query. Respond as if answering directly. If results are empty or indicate 'no results', state that clearly.

            Database Results:
            {results}

            Original Sub-Query:
            {question}

            Answer:""")
        ])
        kg_generate_final_chain = kg_generate_final_prompt | llm | StrOutputParser()
        print("KG GenerateFinalAnswer chain initialized.")

    else:
        print("Warning: KG components could not be initialized due to missing LLM, Embeddings, or KG Neo4j connection.")
        # Set chains to None to prevent errors later
        kg_text2cypher_chain = None
        kg_validate_cypher_chain = None
        kg_cypher_query_corrector = None
        kg_correct_cypher_chain = None
        kg_generate_final_chain = None
        kg_example_selector = None

except Exception as e:
    logging.exception("Failed to connect to KG Neo4j or initialize KG components.")
    print(f"KG NEO4J or COMPONENT INITIALIZATION FAILED: {e}")
    kg_neo4j_graph = None
    # Set chains to None
    kg_text2cypher_chain = None
    kg_validate_cypher_chain = None
    kg_cypher_query_corrector = None
    kg_correct_cypher_chain = None
    kg_generate_final_chain = None
    kg_example_selector = None


# --- NLP Setup (Needed for RAG) ---
# (Keep the spaCy and TF-IDF setup as it was)
nlp = None
if spacy:
    try:
        nlp = spacy.load("en_core_web_sm")
        print("spaCy model loaded.")
    except OSError:
        logging.error("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
        print("spaCy model 'en_core_web_sm' not found. Keyword extraction will fail.")
        nlp = None
else:
    print("spaCy not installed, skipping model loading.")

# ==============================================================================
# == RAG Pipeline Definitions (Using RAG-specific Neo4j)
# ==============================================================================
# (Keep the RAGState and RAG nodes exactly as they were in the original code)
class RAGState(TypedDict):
    question: str
    query_embedding: Optional[List[float]]
    retrieved_chunks: Optional[List[Dict]]
    context: Optional[str]
    final_answer: Optional[BaseMessage]
    final_chunk_metadata: Optional[List[Dict]]

# --- RAG Nodes --- (Keep all RAG nodes as they were)
def embed_query_node(state: RAGState) -> RAGState:
    print("--- RAG: Embedding Query ---")
    if not SentenceTransformer:
        print("Error: SentenceTransformer not available.")
        state["query_embedding"] = None
        return state
    model = SentenceTransformer("all-MiniLM-L6-v2")
    state["query_embedding"] = model.encode(state["question"]).tolist()
    return state

def semantic_and_multi_hop_node(state: RAGState) -> RAGState:
    print("--- RAG: Semantic & Multi-Hop Retrieval ---")
    graph = rag_neo4j_graph
    if not graph or not state.get("query_embedding"):
        print("Warning: Skipping RAG semantic node due to missing RAG graph or embedding.")
        state["retrieved_chunks"] = state.get("retrieved_chunks", [])
        return state
    # (Rest of semantic_and_multi_hop_node logic remains the same)
    base_chunk_ids = []
    try:
        vector_query = """
        CALL db.index.vector.queryNodes('chunk_vector', 3, $embedding)
        YIELD node, score
        RETURN node.chunk_id AS chunk_id
        """
        top_chunks = graph.query(vector_query, params={"embedding": state["query_embedding"]})
        base_chunk_ids = [r["chunk_id"] for r in top_chunks]
        print(f"Semantic top chunks: {base_chunk_ids}")
    except Exception as e:
        logging.warning(f"RAG: Error during vector query: {e}")

    all_related = set(base_chunk_ids)
    if base_chunk_ids:
        try:
            neighbor_query = """
            MATCH (c:Chunk {chunk_id: $cid})-[:PRECEDES|SHARES_KEYWORD|SEMANTIC_SIM]-(n:Chunk)
            WHERE c.chunk_id <> n.chunk_id
            RETURN DISTINCT n.chunk_id AS chunk_id
            LIMIT 2
            """
            for cid in base_chunk_ids:
                neighbors = graph.query(neighbor_query, params={"cid": cid})
                neighbor_ids = [r["chunk_id"] for r in neighbors]
                if neighbor_ids: print(f"Neighbors for {cid}: {neighbor_ids}")
                all_related.update(neighbor_ids)
        except Exception as e:
            logging.warning(f"RAG: Error during neighbor query: {e}")

    existing_chunks = state.get("retrieved_chunks", [])
    existing_ids = {c["chunk_id"] for c in existing_chunks}
    combined_ids = existing_ids.union(all_related)
    state["retrieved_chunks"] = [{"chunk_id": cid} for cid in combined_ids]
    print(f"Semantic+Hop combined chunk IDs: {combined_ids}")
    return state


# --- Keyword functions (Keep as they were) ---
def extract_keywords_from_query(query: str, top_n: int = 10) -> List[str]:
    if not nlp: return []
    if not TfidfVectorizer: return []
    doc = nlp(query.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha and len(token.lemma_) > 2]
    if not tokens: return []
    vectorizer = TfidfVectorizer(max_features=top_n, stop_words='english')
    try:
        vectorizer.fit_transform([" ".join(tokens)])
        return list(vectorizer.get_feature_names_out())
    except ValueError:
        return []

def expand_keywords_via_similar_to(graph: Neo4jGraph, keywords: List[str]) -> List[str]:
    if not graph or not keywords: return keywords
    query = """
    UNWIND $keywords AS kw
    MATCH (:Keyword {name: kw})-[:SIMILAR_TO]-(similar:Keyword)
    RETURN DISTINCT similar.name AS name
    """
    try:
        results = graph.query(query, params={"keywords": keywords})
        return list(set(keywords) | {r["name"] for r in results})
    except Exception as e:
        logging.warning(f"RAG: Error expanding keywords: {e}")
        return keywords

def find_chunks_by_keyword_overlap(graph: Neo4jGraph, keywords: List[str], top_k: int = 2) -> List[str]:
    if not graph or not keywords: return []
    query = """
    UNWIND $keywords AS kw
    MATCH (c:Chunk)-[:HAS_KEYWORD]->(k:Keyword {name: kw})
    WITH c, count(DISTINCT k.name) AS overlap_count
    RETURN c.chunk_id AS chunk_id
    ORDER BY overlap_count DESC
    LIMIT $top_k
    """
    try:
        results = graph.query(query, params={"keywords": keywords, "top_k": top_k})
        return [r["chunk_id"] for r in results]
    except Exception as e:
        logging.warning(f"RAG: Error finding chunks by keyword overlap: {e}")
        return []

def expand_with_precedes(graph: Neo4jGraph, chunk_ids: List[str]) -> List[str]:
    if not graph or not chunk_ids: return chunk_ids
    all_ids = set(chunk_ids)
    for cid in chunk_ids:
        query = """
        MATCH (p:Chunk)-[:PRECEDES]->(c:Chunk {chunk_id: $cid})
        RETURN p.chunk_id AS chunk_id
        UNION
        MATCH (c:Chunk {chunk_id: $cid})-[:PRECEDES]->(n:Chunk)
        RETURN n.chunk_id AS chunk_id
        """
        try:
            results = graph.query(query, params={"cid": cid})
            all_ids.update([r["chunk_id"] for r in results])
        except Exception as e:
            logging.warning(f"RAG: Error expanding with precedes for chunk {cid}: {e}")
            continue
    return list(all_ids)

# --- RAG Keyword Node (Keep as it was) ---
def keyword_and_precedes_node(state: RAGState) -> RAGState:
    print("--- RAG: Keyword & Precedes Retrieval ---")
    graph = rag_neo4j_graph
    if not graph or not nlp or not TfidfVectorizer:
        print("Warning: Skipping RAG keyword node due to missing RAG graph, nlp, or TfidfVectorizer.")
        return state
    # (Rest of keyword_and_precedes_node logic remains the same)
    keywords = extract_keywords_from_query(state["question"])
    print(f"Initial keywords: {keywords}")
    if not keywords:
         print("No keywords extracted.")
         return state

    expanded_kws = expand_keywords_via_similar_to(graph, keywords)
    print(f"Expanded keywords: {expanded_kws}")
    top_chunks_kw = find_chunks_by_keyword_overlap(graph, expanded_kws)
    print(f"Keyword top chunks: {top_chunks_kw}")
    if not top_chunks_kw:
        print("No chunks found via keywords.")
        return state

    expanded_ids_prec = expand_with_precedes(graph, top_chunks_kw)
    print(f"Keyword+Precedes expanded chunk IDs: {expanded_ids_prec}")

    existing_chunks = state.get("retrieved_chunks", [])
    existing_ids = {c["chunk_id"] for c in existing_chunks}
    combined_ids = existing_ids.union(expanded_ids_prec)
    state["retrieved_chunks"] = [{"chunk_id": cid} for cid in combined_ids]
    print(f"Keyword+Precedes combined chunk IDs: {combined_ids}")
    return state

# --- RAG Fetch Texts Node (Keep as it was) ---
def fetch_chunk_texts_node(state: RAGState) -> RAGState:
    print("--- RAG: Fetching Chunk Texts ---")
    graph = rag_neo4j_graph
    retrieved_chunks = state.get("retrieved_chunks", [])
    if not graph or not retrieved_chunks:
        print("Warning: Skipping fetch chunk texts due to missing RAG graph or chunks.")
        state["context"] = ""
        return state
    # (Rest of fetch_chunk_texts_node logic remains the same)
    chunk_ids = [c["chunk_id"] for c in retrieved_chunks]
    print(f"Fetching text for Chunk IDs: {chunk_ids}")

    try:
        result = graph.query("""
            MATCH (c:Chunk)
            WHERE c.chunk_id IN $chunk_ids
            RETURN c.chunk_id AS chunk_id, c.full_text AS text, coalesce(c.text_preview, '') AS preview, coalesce(c.source, 'Unknown Source') AS source
        """, params={"chunk_ids": chunk_ids})
    except Exception as e:
        logging.exception(f"RAG: Error fetching chunk texts: {e}")
        result = []

    id_map = {r["chunk_id"]: r for r in result}
    state["retrieved_chunks"] = [id_map[cid] for cid in chunk_ids if cid in id_map and id_map[cid].get("text")]

    if not state["retrieved_chunks"]:
        print("Warning: No chunk texts were successfully fetched.")
        state["context"] = ""
        return state

    state["context"] = "\n\n---\n\n".join([
        f"📄 Chunk ID: {c.get('chunk_id', 'N/A')} | Source: {c.get('source', 'N/A')}\n{c.get('text', '')}"
        for c in state["retrieved_chunks"]
    ])
    return state

# --- RAG Rerank Node (Keep as it was) ---
def rerank_chunks_node(state: RAGState, top_k: int = 5) -> RAGState:
    print(f"--- RAG: Re-ranking Chunks (Top {top_k}) ---")
    retrieved_chunks = state.get("retrieved_chunks", [])
    question = state.get("question", "")
    if not retrieved_chunks or not question or not llm: # Check for LLM too
        print("Warning: Skipping re-ranking due to missing chunks, question, or LLM.")
        return state
    if len(retrieved_chunks) <= top_k:
        print("Number of chunks is less than or equal to top_k, skipping re-ranking.")
        return state
    # (Rest of rerank_chunks_node logic remains the same)
    formatted_chunks = "\n\n".join([
        f"Chunk ID: {c.get('chunk_id', 'N/A')} | Source: {c.get('source', 'N/A')}\nText: {c.get('text', '')[:500]}"
        for c in retrieved_chunks
    ])

    prompt = ChatPromptTemplate.from_template("""You are an intelligent document relevance ranker.
    Given a user question and a list of text chunks (with IDs and Sources), identify and list the **Top {top_k}** most relevant chunks.
    Output *only* the IDs of the most relevant chunks, one per line, in the format: Chunk ID: <id>
    Do not add explanation or preamble.

    Question:
    {question}

    Available Chunks:
    {chunks}

    Top {top_k} Relevant Chunk IDs:""")

    reranker_llm = llm # Use the globally defined LLM
    chain = prompt | reranker_llm | StrOutputParser()
    try:
        raw_output = chain.invoke({"question": question, "chunks": formatted_chunks, "top_k": top_k})
        print(f"Re-ranker LLM Raw Output:\n{raw_output}")
        matched_ids = re.findall(r"Chunk ID:\s*(.+)", raw_output)
        selected_ids = matched_ids[:top_k]
        print(f"\nSelected IDs after re-ranking ({len(selected_ids)} found): {selected_ids}")

        if selected_ids:
            id_map = {str(c["chunk_id"]): c for c in retrieved_chunks}
            # Ensure the lookup uses the potentially stripped ID if you added .strip() above
            state["retrieved_chunks"] = [id_map[cid] for cid in selected_ids if cid in id_map]
            print(f"Kept {len(state['retrieved_chunks'])} chunks after re-ranking.")
        else:
            print("Warning: Re-ranker did not output valid Chunk IDs. Keeping all chunks before re-ranking.")

        # Make sure the final context is rebuilt based on the *reranked* chunks
        # This requires modifying the generate_answer_node slightly or passing the reranked chunks explicitly.
        # Simplest fix might be to update the context *after* reranking:

        if state.get("retrieved_chunks"): # If reranking didn't fail completely
            state["context"] = "\n\n---\n\n".join([
                f"📄 Chunk ID: {c.get('chunk_id', 'N/A')} | Source: {c.get('source', 'N/A')}\n{c.get('text', '')}"
                for c in state["retrieved_chunks"]
            ])
        else:
            # Decide fallback behavior - maybe keep original context but log a warning?
            print("Warning: Reranking resulted in zero chunks, keeping original context for RAG generation.")
            # state["context"] remains unchanged in this case

    except Exception as e:
        logging.exception(f"Error during chunk re-ranking: {e}")
        print(f"Error during chunk re-ranking: {e}. Keeping all chunks before re-ranking.")
    return state

# --- RAG Generate Answer Node (Keep as it was) ---
def generate_answer_node(state: RAGState) -> RAGState:
    print("--- RAG: Generating Final Answer ---")
    context = state.get("context", "")
    question = state.get("question", "")
    # These are the chunks *after* reranking used to build the context
    retrieved_chunks = state.get("retrieved_chunks", [])

    final_metadata = [] # Initialize list for detailed metadata

    if not context or not question or not llm:
        print("Warning: Skipping final answer generation due to missing context, question, or LLM.")
        state["final_answer"] = SystemMessage(content="Could not generate answer due to missing context, question, or LLM.")
        state["final_chunk_metadata"] = [] # Empty metadata on failure
        return state

    # --- CHANGE: REMOVE source_info string ---
    # source_info = ""
    # if retrieved_chunks:
    #     source_list = "\n".join(f"- Chunk ID: {c.get('chunk_id', 'N/A')} | Source: {c.get('source', 'N/A')}" for c in retrieved_chunks)
    #     source_info = f"\n\nSources Used:\n{source_list}"
    # --- END CHANGE ---

    # --- Populate final_chunk_metadata ---
    if retrieved_chunks:
         final_metadata = [
             {"chunk_id": c.get("chunk_id", "N/A"), "source_doc": c.get("source", "N/A")}
             for c in retrieved_chunks
         ]
         print(f"Storing metadata for {len(final_metadata)} final chunks.")
    # --- End Population ---


    # --- CHANGE: Simplified prompt - Don't ask LLM to append source info ---
    prompt = ChatPromptTemplate.from_template("""You are a helpful question-answering assistant.
    Answer the user's question based *only* on the provided context. Be concise.
    If the context doesn't contain the answer, state that clearly. Do *not* add information not present in the context.

    Context:
    {context}
    ---
    Question: {question}
    ---
    Answer:""")
    # --- END CHANGE ---

    chain = prompt | llm # Output will be an AIMessage
    try:
        # --- CHANGE: Remove source_info from invoke ---
        llm_response: BaseMessage = chain.invoke({"question": question, "context": context})
        state["final_answer"] = llm_response
        # --- END CHANGE ---
        state["final_chunk_metadata"] = final_metadata # Store the collected metadata
        print(f"RAG Final Answer Generated (type: {type(llm_response)}). Content preview: {llm_response.content[:100]}...")
    except Exception as e:
        logging.exception(f"Error during final answer generation: {e}")
        print(f"Error during final answer generation: {e}")
        state["final_answer"] = SystemMessage(content=f"Error generating final answer: {e}")
        state["final_chunk_metadata"] = [] # Clear metadata on error

    return state

# --- RAG Graph Workflow Definition (Keep as it was) ---
rag_workflow = StateGraph(RAGState)
rag_workflow.add_node("embed_query", embed_query_node)
rag_workflow.add_node("semantic_and_multi_hop", semantic_and_multi_hop_node)
rag_workflow.add_node("keyword_and_precedes", keyword_and_precedes_node)
rag_workflow.add_node("fetch_chunk_texts", fetch_chunk_texts_node)
rag_workflow.add_node("rerank_chunks", rerank_chunks_node)
rag_workflow.add_node("generate_answer", generate_answer_node)
rag_workflow.set_entry_point("embed_query")
rag_workflow.add_edge("embed_query", "semantic_and_multi_hop")
rag_workflow.add_edge("semantic_and_multi_hop", "keyword_and_precedes")
rag_workflow.add_edge("keyword_and_precedes", "fetch_chunk_texts")
rag_workflow.add_edge("fetch_chunk_texts", "rerank_chunks")
rag_workflow.add_edge("rerank_chunks", "generate_answer")
rag_workflow.add_edge("generate_answer", END)

# Compile the RAG graph
rag_graph = rag_workflow.compile()
print("RAG Graph compiled.")


# ==============================================================================
# == Main Pipeline Definitions (Using Updated OverallState)
# ==============================================================================

# --- OverallState Definition (Updated) ---
class InputState(TypedDict):
    question: str

class OverallState(TypedDict):
    question: str
    next_action: str # Used by guardrails
    # Removed KG-specific fields: cypher_statement, cypher_errors, database_records
    guardrail_message: Optional[str] # To store guardrail end message
    steps: Annotated[List[str], add]
    subqueries: List[str]
    routes: Dict[str, str]
    # Collect answers here {subquery: str, answer: str, source: str}
    subquery_answers: Annotated[List[Dict], add]
    final_answer: str
    final_source_details_str: str # <<< ADDED: Formatted string for final display

# --- Guardrails Node (Keep as it was) ---
guardrails_system = """As an intelligent assistant, decide if a question relates to hospital/surgery info or healthcare implementation documents.
If related to hospitals, surgeries, specialties, empanelment, states, districts, or key implementation guidelines (HI, Anti-Fraud, Beneficiary ID/Empowerment, Hospital Empanelment/De-Empanelment, IEC), output "hospital".
Otherwise, output "end".
Return ONLY the JSON structure matching GuardrailsOutput.

Relevant Guidelines:
- HI (Health Information) Guidelines
- Anti-Fraud Guidelines
- Beneficiary Identification Guidelines
- Beneficiary Empowerment Guidebook
- Hospital Empanelment and De-Empanelment Guidelines
- IEC (Information, Education, Communication) Guidebook
"""
guardrails_prompt = ChatPromptTemplate.from_messages([
    ("system", guardrails_system),
    ("human", ("{question}")),
])
class GuardrailsOutput(BaseModel):
    decision: Literal["hospital", "end"]
guardrails_chain = guardrails_prompt | llm.with_structured_output(GuardrailsOutput)

def guardrails(state: InputState) -> OverallState:
    """ Decides if the question is relevant. Initializes OverallState. """
    print("--- Running Guardrails ---")
    question = state.get("question")
    if not question or not llm: # Check LLM
        print("Error: Question not found or LLM not initialized.")
        return { "question": "", "next_action": "end", "guardrail_message": "Error: Input question was missing or LLM failed.",
                 "steps": ["guardrail (error)"], "subqueries": [], "routes": {}, "subquery_answers": [], "final_answer": "Error: Input question missing or LLM failed."}

    decision = "hospital"
    guardrail_message = None
    try:
        guardrails_output: GuardrailsOutput = guardrails_chain.invoke({"question": question})
        decision = guardrails_output.decision
        print(f"Guardrails decision: {decision}")
        if decision == "end":
            guardrail_message = "This question is not about hospitals, surgeries, specialties, empanelment, states, districts, or key healthcare implementation documents. I cannot answer this question."
    except Exception as e:
        logging.exception(f"Error invoking/parsing guardrails chain: {e}")
        print(f"Error during guardrails execution: {e}. Defaulting to 'hospital'.")
        decision = "hospital"

    initialized_state: OverallState = {
        "question": question, "next_action": decision, "guardrail_message": guardrail_message,
        "steps": ["guardrail"], "subqueries": [], "routes": {}, "subquery_answers": [], "final_answer": "", "final_source_details_str": "",
    }
    return initialized_state

def guardrails_condition(state: OverallState) -> Literal["multi_query_split_node", "final_answer_node"]:
    """ Routes based on guardrails decision. """
    print(f"--- Guardrails Condition: next_action = {state.get('next_action')} ---")
    return "multi_query_split_node" if state.get("next_action") == "hospital" else "final_answer_node"

# --- Multi-Query Split Node (Keep as it was) ---
class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                parts = stripped.split(". ", 1)
                cleaned.append(parts[1] if len(parts) == 2 and parts[0].isdigit() else stripped.lstrip("- >* "))
        return cleaned

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an assistant breaking down a complex question into 3 distinct sub-questions for retrieval.
    Produce 3 relevant sub-queries. Avoid excessive specificity unless the original question demands it. Phrase as clear questions.
    Keep terms simple if the original uses simple terms. Generate 3 sub-queries.

    Original question: {question}

    Sub-queries: (Output one question per line, no numbering)"""
)
def multi_query_split_node(state: OverallState) -> Dict[str, Any]:
    """ Generates sub-queries. """
    print("--- Running Multi-Query Split ---")
    question = state.get("question")
    if not question or not llm: # Check LLM
        print("Warning: No question or LLM for multi-query split.")
        return {"subqueries": [], "steps": ["split (error)"]}

    parser = LineListOutputParser()
    chain = QUERY_PROMPT | llm | parser
    subqueries = []
    try:
        subqueries = chain.invoke({"question": question})
        if not isinstance(subqueries, list) or not all(isinstance(q, str) for q in subqueries):
             print(f"Warning: Multi-query split bad format. Output: {subqueries}")
             subqueries = []
        subqueries = [q for q in subqueries if q][:3] # Filter empty strings and limit
        print(f"Generated Subqueries: {subqueries}")
    except Exception as e:
        logging.exception(f"Error during multi-query split: {e}")
        subqueries = []

    return {"subqueries": subqueries, "steps": ["split"]}

# --- Routing Node (Keep as it was) ---
routing_system_prompt = """You are an intelligent query router. Route to 'vectordb' or 'knowledge_db'.

## Data Sources:
1.  **Vector Database (`vectordb`)**: Contains text from healthcare guidelines (Anti-Fraud, Beneficiary ID/Empowerment, Hospital Empanelment/De-Empanelment, IEC). Use for procedures, steps, guidelines, eligibility, policy, cost rules.
    *   Examples: "Steps for hospital empanelment?", "Explain beneficiary identification process.", "IEC policy?"
2.  **Knowledge Graph (`knowledge_db`)**: Contains structured data: Hospitals (ID, location, contact, empanelment), Surgeries (codes, types, cost), States, Districts, Specialties. Use for specific entity lookups.
    *   Examples: "Hospitals in Tamil Nadu?", "Cost of surgery XYZ?", "Specialties in Bihar?"

## Logic:
- If query is about **steps, procedures, process, eligibility, rules, guidelines** -> `vectordb`
- If query asks for specific **lists of hospitals, surgeries, counts, locations, statuses** -> `knowledge_db`
- Ignore location keywords unless they change the *intent* (e.g., "Guidelines *in Tamil Nadu*" -> vectordb, "Hospitals *in Tamil Nadu*" -> knowledge_db).
- If unsure or mixed, prefer `vectordb` if guidelines/process is mentioned.

Return ONLY 'vectordb' or 'knowledge_db' using the RouteQuery JSON schema."""

class RouteQuery(BaseModel):
    datasource: Literal["vectordb", "knowledge_db"] = Field(description="Select 'vectordb' for processes/guidelines, 'knowledge_db' for specific entities.")
def create_router_chain(llm_instance: Runnable) -> Runnable:
    router_prompt = ChatPromptTemplate.from_messages([("system", routing_system_prompt),("human", "{question}")])
    return router_prompt | llm_instance.with_structured_output(RouteQuery)

def routing_node(state: OverallState) -> Dict[str, Any]:
    """ Routes each subquery. """
    print("--- Running Routing ---")
    subqueries = state.get("subqueries", [])
    if not subqueries or not llm: # Check LLM
        print("Warning: No subqueries or LLM for routing.")
        return {"routes": {}, "steps": ["route (skipped)"]}

    router_chain = create_router_chain(llm)
    routes = {}
    try:
        for subquery in subqueries:
            if not subquery or not isinstance(subquery, str): continue
            print(f"Routing subquery: '{subquery}'")
            result: RouteQuery = router_chain.invoke({"question": subquery})
            routes[subquery] = result.datasource
            print(f"--> Route: {result.datasource}")
            # time.sleep(1) # Optional delay
    except Exception as e:
        logging.exception(f"Error during routing: {e}")
        print(f"Error during routing node: {e}. Some routes may be missing.")

    return {"routes": routes, "steps": ["route"]}


# ==============================================================================
# == KG Query Execution Logic (Refactored from KG code)
# ==============================================================================

# Max validation/correction attempts
MAX_KG_ATTEMPTS = 2
KG_NO_RESULTS_MSG = "I couldn't find relevant information for this specific query in the knowledge graph."

def run_kg_query_pipeline(subquery: str, graph: Neo4jGraph, llm_instance: Runnable) -> Dict[str, str]:
    """
    Executes the full KG pipeline (generate, validate, correct, execute, format) for a single subquery.
    Returns a dictionary {"answer": str, "source": str}.
    """
    print(f"\n--- Running KG Pipeline for Subquery: '{subquery}' ---")

    # Check if necessary components are initialized
    if not all([graph, llm_instance, kg_text2cypher_chain, kg_validate_cypher_chain,
                kg_cypher_query_corrector, kg_correct_cypher_chain, kg_generate_final_chain, kg_example_selector]):
        print("Error: KG components not initialized.")
        return {"answer": "Error: Knowledge Graph components not available.", "source": "Knowledge Graph (Error)"}

    cypher = ""
    errors = []
    records = None
    final_kg_answer = ""
    attempt = 0

    # --- Get Schema and Examples ---
    try:
        schema = graph.schema
        NL = "\n"
        fewshot_examples_str = (NL * 2).join(
            [f"Question: {el['question']}{NL}Cypher:{el['query']}"
             for el in kg_example_selector.select_examples({"question": subquery})]
        )
    except Exception as e:
        print(f"Error getting schema or examples: {e}")
        return {"answer": f"Error preparing KG query: {e}", "source": "Knowledge Graph (Error)"}

    while attempt < MAX_KG_ATTEMPTS:
        attempt += 1
        print(f"\nKG Attempt {attempt}/{MAX_KG_ATTEMPTS}")

        # --- 1. Generate Cypher (or use corrected from previous loop) ---
        if not cypher: # Only generate on first attempt
            try:
                print("Generating Cypher...")
                cypher = kg_text2cypher_chain.invoke({
                    "question": subquery,
                    "fewshot_examples": fewshot_examples_str,
                    "schema": schema,
                })
                print(f"Generated Cypher: {cypher}")
                # Basic check if it's empty or placeholder
                if not cypher or "cannot construct cypher" in cypher.lower():
                    raise ValueError("LLM failed to generate valid Cypher.")
            except Exception as e:
                print(f"Error generating Cypher: {e}")
                final_kg_answer = f"Error generating database query: {e}"
                break # Exit loop on generation error

        # --- 2. Validate Cypher ---
        print("Validating Cypher...")
        errors = []
        mapping_errors = []
        corrected_cypher = cypher # Start with current cypher

        # a) Syntax Check
        try:
            graph.query(f"EXPLAIN {cypher}")
            print("Syntax OK.")
        except CypherSyntaxError as e:
            print(f"Syntax Error: {e.message}")
            errors.append(f"Syntax Error: {e.message}")
        except Exception as e: # Catch other potential query errors during EXPLAIN
             print(f"EXPLAIN Error: {e}")
             errors.append(f"EXPLAIN Error: {e}")


        # b) Corrector (Relationship Direction - Experimental)
        try:
            # Check if corrector is initialized
            if kg_cypher_query_corrector:
                corrected_cypher_rel = kg_cypher_query_corrector(cypher)
                if not corrected_cypher_rel:
                     errors.append("Cypher statement doesn't fit schema (corrector failed).")
                     print("Corrector: Schema fit failed.")
                elif corrected_cypher_rel != cypher:
                     print("Corrector: Relationship direction potentially corrected.")
                     corrected_cypher = corrected_cypher_rel # Use corrected version
                else:
                     print("Corrector: No relationship direction changes needed.")
            else:
                 print("Corrector: Skipped (not initialized).")
        except Exception as e:
            print(f"Error during relationship correction: {e}")
            errors.append(f"Error during relationship correction: {e}")


        # c) LLM Validation (Semantic, Schema adherence, Value Mapping)
        try:
            llm_validation: ValidateCypherOutput = kg_validate_cypher_chain.invoke({
                "question": subquery,
                "schema": schema,
                "cypher": corrected_cypher, # Validate the potentially corrected cypher
            })
            if llm_validation.errors:
                print(f"LLM Validation Errors: {llm_validation.errors}")
                errors.extend(llm_validation.errors)

            # Check value mapping if filters were identified
            if llm_validation.filters:
                print("Checking filter value mappings...")
                for filter_prop in llm_validation.filters:
                    # Check if property exists and is STRING type in schema (simple check)
                    prop_schema = next((p for p in graph.structured_schema.get("node_props", {}).get(filter_prop.node_label, [])
                                        if p["property"] == filter_prop.property_key), None)

                    if prop_schema and prop_schema["type"] == "STRING":
                        try:
                            mapping = graph.query(
                                f"MATCH (n:{graph.sanitize_node_label(filter_prop.node_label)}) " # Sanitize label
                                f"WHERE toLower(n.`{graph.sanitize_property_key(filter_prop.property_key)}`) = toLower($value) " # Sanitize key
                                "RETURN 'exists' LIMIT 1",
                                {"value": filter_prop.property_value},
                            )
                            if not mapping:
                                map_err = f"Value '{filter_prop.property_value}' not found for property '{filter_prop.property_key}' on node '{filter_prop.node_label}'."
                                print(f"Mapping Error: {map_err}")
                                mapping_errors.append(map_err)
                            else:
                                print(f"Mapping OK: {filter_prop.property_value} found for {filter_prop.node_label}.{filter_prop.property_key}")
                        except Exception as map_e:
                             map_err = f"Error checking mapping for {filter_prop.node_label}.{filter_prop.property_key}='{filter_prop.property_value}': {map_e}"
                             print(f"Mapping Check Error: {map_err}")
                             mapping_errors.append(map_err) # Treat query error during mapping as a mapping error
                    else:
                        print(f"Skipping mapping check for non-string or unknown property: {filter_prop.node_label}.{filter_prop.property_key}")
        except Exception as e:
            print(f"Error during LLM validation/mapping: {e}")
            errors.append(f"LLM validation failed: {e}")

        # --- 3. Decide Next Step ---
        if mapping_errors:
            print("Mapping errors found. Cannot execute query.")
            final_kg_answer = f"Could not answer query. Specific values mentioned might not exist in the knowledge base: {'; '.join(mapping_errors)}"
            break # Exit loop

        if errors:
            print("Errors found. Attempting correction...")
            # --- 4. Correct Cypher ---
            if attempt < MAX_KG_ATTEMPTS:
                try:
                    corrected_cypher = kg_correct_cypher_chain.invoke({
                        "question": subquery,
                        "errors": "\n".join(errors),
                        "cypher": corrected_cypher, # Pass the cypher that failed validation
                        "schema": schema,
                    })
                    print(f"Corrected Cypher: {corrected_cypher}")
                    cypher = corrected_cypher # Use corrected cypher for next validation attempt
                    errors = [] # Clear errors for the next validation attempt
                    continue # Go back to validation
                except Exception as e:
                    print(f"Error correcting Cypher: {e}")
                    final_kg_answer = f"Failed to correct database query: {e}"
                    break # Exit loop on correction error
            else:
                print("Max correction attempts reached.")
                final_kg_answer = f"Could not execute query after multiple correction attempts. Errors: {'; '.join(errors)}"
                break # Exit loop

        else:
            # --- 5. Execute Cypher ---
            print("Validation successful. Executing Cypher...")
            try:
                records = graph.query(corrected_cypher) # Use the final validated/corrected cypher
                print(f"Execution successful. Records retrieved: {len(records) if records else 0}")
                # Format final answer using LLM
                if not records:
                     final_kg_answer = KG_NO_RESULTS_MSG
                else:
                    # Use the specific KG final answer chain
                    final_kg_answer = kg_generate_final_chain.invoke({
                        "question": subquery,
                        "results": records
                    })
                break # Exit loop successfully

            except Exception as e:
                print(f"Error executing Cypher: {e}")
                final_kg_answer = f"Error executing database query: {e}"
                break # Exit loop on execution error

    print(f"--- Finished KG Pipeline for Subquery. Final Answer: {final_kg_answer[:100]}... ---")
    return {"answer": final_kg_answer, "source": "Knowledge Graph"}


# ------------------------
# Step 3a: Knowledge Graph Answer Node (Updated)
# ------------------------
def kg_answer_node(state: OverallState) -> Dict[str, Any]:
    """ Retrieves answers for subqueries routed to 'knowledge_db'. """
    print("--- Running Knowledge Graph Retrieve ---")
    # ... (error checking) ...

    new_answers = []
    kg_invoked = False
    subqueries = state.get("subqueries", [])
    routes = state.get("routes", {})

    # ... (check KG components) ...
    # Add error messages for subqueries routed here if needed, or just skip
    for q in subqueries:
        if routes.get(q) == "knowledge_db":
            kg_invoked = True # Mark as invoked even if failed
            new_answers.append({
                "subquery": q,
                "answer": "Knowledge Graph query skipped due to connection/LLM issue.",
                "source_type": "Knowledge Graph", # Use consistent type name
                "details": {"error": "KG components not available"} # Store details
            })
    # ...

    # Process subqueries routed to knowledge_db
    for q in subqueries:
        if routes.get(q) == "knowledge_db":
            kg_invoked = True
            kg_result = run_kg_query_pipeline(q, kg_neo4j_graph, llm)

            # --- CHANGE: Append result with new structure ---
            new_answers.append({
                "subquery": q,
                "answer": kg_result.get("answer", "Error retrieving KG answer."),
                "source_type": "Knowledge Graph", # Consistent type
                "details": {"info": f"KG pipeline executed for subquery."} if "Error" not in kg_result.get("source", "") else {"error": kg_result.get("answer")} # Simple detail for KG
            })
            # --- END CHANGE ---

    step_name = "kg_retrieve (invoked KG)" if kg_invoked else "kg_retrieve (skipped)"
    # Return the collected answers to be added to the main state
    return {"subquery_answers": new_answers, "steps": [step_name]}


# ------------------------
# Step 3b: RAG Answer Node (Keep as it was)
# ------------------------
def rag_answer_node(state: OverallState) -> Dict[str, Any]:
    """ Invokes the RAG graph for subqueries routed to 'vectordb'. """
    print("--- Running RAG Retrieve Node (Invoking RAG Graph) ---")
    # ... (error checking for graph/llm) ...

    new_answers = []
    rag_invoked = False
    subqueries = state.get("subqueries", [])
    routes = state.get("routes", {})

    for subquery in subqueries:
        if routes.get(subquery) == "vectordb":
            print(f"\n--- Invoking RAG graph for subquery: '{subquery}' ---")
            rag_invoked = True
            try:
                rag_input: RAGState = {"question": subquery}
                rag_result: RAGState = rag_graph.invoke(rag_input) # Invoke RAG graph

                final_answer_msg = rag_result.get("final_answer")
                # --- CHANGE: Get the detailed metadata ---
                final_metadata = rag_result.get("final_chunk_metadata", [])
                # --- END CHANGE ---

                if final_answer_msg and hasattr(final_answer_msg, 'content'):
                    # --- CHANGE: Answer text is now clean ---
                    answer_text = final_answer_msg.content
                    print(f"RAG graph successful. Answer preview: {answer_text[:150]}...")
                    new_answers.append({
                        "subquery": subquery,
                        "answer": answer_text, # Clean answer text
                        "source_type": "VectorDB", # Use a consistent type name
                        "details": final_metadata # Store the list of chunk metadata
                    })
                    # --- END CHANGE ---
                else:
                    # ... (handle no valid answer) ...
                    # --- CHANGE: Store appropriate structure on failure ---
                    new_answers.append({
                        "subquery": subquery,
                        "answer": "Failed to retrieve detailed answer from RAG pipeline.",
                        "source_type": "VectorDB",
                        "details": [{"error": "No valid answer message"}] # Indicate failure
                    })
                    # --- END CHANGE ---
            except Exception as e:
                 # ... (handle exception) ...
                 # --- CHANGE: Store appropriate structure on exception ---
                 new_answers.append({
                     "subquery": subquery,
                     "answer": f"Error during RAG pipeline execution: {e}",
                     "source_type": "VectorDB",
                     "details": [{"error": str(e)}] # Indicate failure
                 })
                 # --- END CHANGE ---
            print(f"--- Finished RAG graph invocation for subquery: '{subquery}' ---\n")

    step_name = "rag_retrieve (invoked RAG)" if rag_invoked else "rag_retrieve (skipped)"
    # --- CHANGE: Return the list with new structure ---
    return {"subquery_answers": new_answers, "steps": [step_name]}
    # --- END CHANGE ---


# ------------------------
# Step 4: Routing Condition (Keep as it was)
# ------------------------
def route_to_kg_or_rag(state: OverallState) -> List[str]:
    """ Determines which answer nodes (KG, RAG) to call. """
    print(f"--- Routing Condition: Checking routes {state.get('routes')} ---")
    targets = []
    routes = state.get("routes", {})
    if isinstance(routes, dict):
        if any(v == "knowledge_db" for v in routes.values()): targets.append("kg_answer_node")
        if any(v == "vectordb" for v in routes.values()): targets.append("rag_answer_node")
    if not targets:
        print("No valid routes found for KG or RAG, proceeding to final answer.")
        return ["final_answer_node"]
    return targets

# ------------------------
# Step 5: Fusion and Final Answer Node (Keep as it was)
# ------------------------
def rrf_fuse_subquery_answers(subquery_answers: List[Dict], k=60) -> List[Document]:
    """ Fuses answers using Reciprocal Rank Fusion (RRF). """
    print("--- Running RRF Fusion ---")
    if not subquery_answers: return []
    # (RRF logic remains the same)
    ranked_lists: List[List[Document]] = []
    for i, ans_data in enumerate(subquery_answers):
        content = ans_data.get("answer")
        if not isinstance(content, str):
            print(f"Warning: Skipping subquery answer due to invalid content: {ans_data}")
            continue
        metadata = { "subquery": ans_data.get("subquery", f"query_{i}"), "source": ans_data.get("source", "unknown") }
        ranked_lists.append([Document(page_content=content, metadata=metadata)])

    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}
    for docs in ranked_lists:
        for rank, doc in enumerate(docs): # rank is always 0 here
            key = doc.page_content
            if key not in doc_map: doc_map[key] = doc
            scores[key] = scores.get(key, 0) + 1.0 / (rank + k)

    sorted_keys = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused_docs = [doc_map[key] for key, _score in sorted_keys]
    print(f"Fused {len(fused_docs)} documents from {len(subquery_answers)} subquery answers.")
    return fused_docs

def final_answer_node(state: OverallState) -> Dict[str, Any]:
    """ Generates the final synthesized answer and the detailed source string. """
    print("--- Running Final Answer Generation ---")
    final_answer_text = ""
    final_source_details_str = "No specific sources identified." # Default
    new_steps = []

    # --- Handle Guardrail End ---
    if state.get("next_action") == "end":
        final_answer_text = state.get("guardrail_message", "Process ended by guardrail.")
        final_source_details_str = "Source: Guardrail decision"
        new_steps = ["final_answer (guardrail end)"]
    # --- Handle LLM Error ---
    elif not llm:
         final_answer_text = "Cannot generate final answer because the LLM is not available."
         final_source_details_str = "Source: LLM Error"
         new_steps = ["final_answer (LLM error)"]
    # --- Process Subquery Answers ---
    else:
        print("Fusing subquery answers and generating final response.")
        sub_answers = state.get("subquery_answers", [])
        if not sub_answers:
             final_answer_text = "I could not retrieve any information to answer the question."
             # Keep default source string
             new_steps = ["fusion (no results)", "final_answer (no context)"]
        else:
            # --- RRF Fusion (Only ranks text, doesn't need source details itself) ---
            # We need the original sub_answers list later for source attribution
            fused_docs = rrf_fuse_subquery_answers(sub_answers) # Uses 'answer' field for content

            if not fused_docs:
                final_answer_text = "I couldn't synthesize an answer after fusing the retrieved information."
                # Keep default source string
                new_steps = ["fusion (empty)", "final_answer (no context)"]
            else:
                # --- Generate Synthesized Answer Text ---
                # Context uses the clean 'answer' text from sub_answers
                context = "\n\n---\n\n".join(
                    f"Source System: [{d.metadata.get('source', 'N/A')}] | Subquery: [{d.metadata.get('subquery', 'N/A')}]\nRetrieved Answer:\n{d.page_content}"
                    for d in fused_docs
                )
                prompt = ChatPromptTemplate.from_messages([
                     ("system", "You are a helpful AI assistant. Synthesize the information from the context (answers to sub-queries) to provide a single, coherent answer to the user's original main question. Base the answer *only* on the provided context. Answer clearly and concisely. Do not explicitly mention 'Source System:' or 'Subquery:' in your final response unless it's naturally part of the information itself."),
                     ("human", "Context:\n{context}\n\nMain Question: {question}\n\nSynthesized Answer:")
                 ])
                chain = prompt | llm | parser
                try:
                    final_answer_text = chain.invoke({"context": context, "question": state["question"]})
                    new_steps = ["fusion", "llm_final_answer"]

                    # --- Build Detailed Source String AFTER successful synthesis ---
                    source_parts = []
                    kg_mentioned = False
                    rag_chunks_by_doc = {} # {doc_source: [chunk_id1, chunk_id2]}

                    # Iterate original sub_answers which have the details
                    for ans_data in sub_answers:
                        source_type = ans_data.get("source_type")
                        details = ans_data.get("details")

                        if source_type == "Knowledge Graph" and not kg_mentioned:
                             source_parts.append("Knowledge Graph")
                             kg_mentioned = True
                        elif source_type == "VectorDB" and isinstance(details, list):
                             for item in details:
                                 if isinstance(item, dict) and "chunk_id" in item: # Check if it's valid metadata
                                     chunk_id = item.get("chunk_id", "N/A")
                                     doc_source = item.get("source_doc", "Unknown Document")
                                     if doc_source not in rag_chunks_by_doc:
                                         rag_chunks_by_doc[doc_source] = set() # Use set for uniqueness
                                     if chunk_id != "N/A":
                                         rag_chunks_by_doc[doc_source].add(chunk_id)

                    # Format RAG details
                    if rag_chunks_by_doc:
                         rag_source_strs = []
                         # Sort documents for consistent output
                         for doc_source in sorted(rag_chunks_by_doc.keys()):
                              # Sort chunk IDs for consistent output
                              chunk_ids = sorted(list(rag_chunks_by_doc[doc_source]))
                              # Limit number displayed if needed
                              max_ids_to_show = 10
                              ids_display = chunk_ids[:max_ids_to_show]
                              ids_str = ", ".join(ids_display)
                              if len(chunk_ids) > max_ids_to_show:
                                   ids_str += f", ... ({len(chunk_ids) - max_ids_to_show} more)"
                              rag_source_strs.append(f"VectorDB ({doc_source}): [{ids_str}]")
                         source_parts.extend(rag_source_strs)

                    final_source_details_str = "; ".join(source_parts) if source_parts else "Sources used, but details not available."
                    # --- End Source String Building ---

                except Exception as e:
                    logging.exception(f"Error during final answer LLM call: {e}")
                    final_answer_text = "Error generating final answer from context."
                    final_source_details_str = "Source: Error during final synthesis"
                    new_steps = ["fusion", "llm_final_answer (error)"]

    print(f"Final Answer Text: {final_answer_text}")
    print(f"Final Source Details String: {final_source_details_str}")

    # --- Return updated state fields ---
    return {
        "final_answer": final_answer_text,
        "steps": new_steps,
        "final_source_details_str": final_source_details_str # Return the formatted string
    }


# --- Build Main Graph (Keep structure as it was) ---
def build_main_pipeline():
    """Builds the main LangGraph pipeline."""
    main_graph = StateGraph(OverallState)

    # Add Nodes
    main_graph.add_node("guardrails", guardrails)
    main_graph.add_node("multi_query_split_node", multi_query_split_node)
    main_graph.add_node("routing_node", routing_node)
    main_graph.add_node("kg_answer_node", kg_answer_node)     # <<< Updated KG node
    main_graph.add_node("rag_answer_node", rag_answer_node)    # RAG node (invokes rag_graph)
    main_graph.add_node("final_answer_node", final_answer_node)

    # Define Edges
    main_graph.set_entry_point("guardrails")
    main_graph.add_conditional_edges("guardrails", guardrails_condition, {
        "multi_query_split_node": "multi_query_split_node",
        "final_answer_node": "final_answer_node"
    })
    main_graph.add_edge("multi_query_split_node", "routing_node")
    main_graph.add_conditional_edges("routing_node", route_to_kg_or_rag, {
        "kg_answer_node": "kg_answer_node",
        "rag_answer_node": "rag_answer_node",
        "final_answer_node": "final_answer_node" # Fallback
    })
    # Edges from KG/RAG nodes converge on the final answer node
    main_graph.add_edge("kg_answer_node", "final_answer_node")
    main_graph.add_edge("rag_answer_node", "final_answer_node")
    main_graph.add_edge("final_answer_node", END)

    print("Compiling main pipeline graph...")
    compiled_graph = main_graph.compile()
    print("Main pipeline graph compiled.")
    return compiled_graph

# --- Example Run ---
# if __name__ == "__main__":
#     print("\n--- Initializing Main Pipeline ---")
#     # Ensure all components (LLM, RAG/KG Neo4j, RAG graph) are initialized above

#     if not llm:
#          print("FATAL: LLM not initialized. Cannot proceed.")
#     elif not rag_neo4j_graph and not kg_neo4j_graph: # Check if at least one DB is available
#          print("FATAL: Neither RAG nor KG Neo4j connection is available. Cannot proceed.")
#     elif not rag_graph:
#          print("FATAL: RAG graph not compiled. Cannot proceed.")
#     else:
#         main_pipeline = build_main_pipeline()

#         print("\n--- Invoking Main Pipeline with Example Question ---")
#         # Example question designed to hit both KG and RAG
#         question = "Which state has the highest number of hospitals and what is the process involved in hospital empanelment?"
#         # Example question designed to hit only KG
#         # question = "How many hospitals are in Tamil Nadu?"
#         # Example question designed to hit only RAG
#         # question = "What are the anti-fraud guidelines?"
#         # Example question for guardrails
#         # question = "What is the weather today?"

#         initial_state: InputState = {"question": question}
#         try:
#             start_time = time.time()
#             final_state = main_pipeline.invoke(initial_state)
#             end_time = time.time()

#             print("\n--- Execution Complete ---")
#             print(f"Time taken: {end_time - start_time:.2f} seconds")
#             print("\nSteps taken:", final_state.get("steps", "N/A"))

#             print("\nSubquery Answers Collected (Internal State):") # Clarify this is intermediate
#             for i, ans in enumerate(final_state.get("subquery_answers", [])):
#                 print(f"  {i+1}. Subquery: {ans.get('subquery')}")
#                 print(f"     Source Type: {ans.get('source_type')}")
#                 # Shorten preview even more
#                 print(f"     Answer Preview: {ans.get('answer', '')[:80]}...")
#                 # Optionally print details preview
#                 # print(f"     Details Preview: {str(ans.get('details'))[:100]}...")

#             print("\n--- Final Synthesized Answer ---")
#             print(final_state.get("final_answer", "No final answer generated."))

#             # <<< CHANGE: Print the detailed source string >>>
#             print("\n--- Sources Used ---")
#             print(final_state.get("final_source_details_str", "No source details generated."))


#         except Exception as e:
#             logging.exception("Error during main pipeline execution:")
#             print(f"\n--- Main Pipeline Execution Failed ---")
#             print(f"Error: {e}")

# --- END OF MODIFIED CODE ---

#modi for flask
from flask import Flask, request, jsonify
import logging
import os

# Initialize Flask app
app = Flask(__name__)

# Ensure all components (LLM, RAG/KG Neo4j, RAG graph) are initialized above
if not llm:
    logging.error("FATAL: LLM not initialized. Cannot proceed.")
elif not rag_neo4j_graph and not kg_neo4j_graph:  # Check if at least one DB is available
    logging.error("FATAL: Neither RAG nor KG Neo4j connection is available. Cannot proceed.")
elif not rag_graph:
    logging.error("FATAL: RAG graph not compiled. Cannot proceed.")
else:
    main_pipeline = build_main_pipeline()
    
@app.route('/status')
def components_status():
    """
    API endpoint to check the status of all components.
    Returns a JSON object indicating the status of each component.
    """
    status = {
        "LLM": "Initialized" if llm else "Not Initialized",
        "RAG_Neo4j_Graph": "Connected" if rag_neo4j_graph else "Not Connected",
        "KG_Neo4j_Graph": "Connected" if kg_neo4j_graph else "Not Connected",
        "RAG_Graph": "Compiled" if rag_graph else "Not Compiled",
        "Main_Pipeline": "Initialized" if 'main_pipeline' in globals() else "Not Initialized"
    }
    return jsonify(status), 200

# Define the API route
@app.route('/api/query', methods=['POST'])
def query_pipeline():
    """
    API endpoint to process a question and return the final state as JSON.
    """
    try:
        # Parse input JSON
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "Question is required"}), 400

        # Prepare initial state
        initial_state: InputState = {"question": question}

        # Invoke the main pipeline
        final_state = main_pipeline.invoke(initial_state)

        # Return the final state as JSON
        return jsonify(final_state), 200

    except Exception as e:
        logging.exception("Error during pipeline execution:")
        return jsonify({"error": str(e)}), 500

# Run the Flask app (for local testing)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)