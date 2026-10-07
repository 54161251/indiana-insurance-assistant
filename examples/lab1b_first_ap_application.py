import streamlit as st
import ollama
import chromadb

documents = [
    "Bloomington is home to Indiana University’s flagship campus, which was founded in 1820.",
    "Bloomington has been designated a 'Tree City USA' for over 30 years because of its commitment to urban forestry.",
    "The city is known for its limestone quarries, which have supplied stone for famous buildings like the Empire State Building.",
    "Bloomington’s population was around 79,000 as of the 2020 U.S. Census, with a significant portion being college students.",
    "Bloomington is surrounded by scenic nature, including Monroe Lake, Indiana’s largest inland lake, and the Hoosier National Forest.",
    "The downtown area features the B-Line Trail, a popular paved path for walking, running, and biking that stretches nearly 4 miles.",
    "Bloomington’s local food scene is celebrated for its international restaurants, reflecting the city’s diverse population.",
    "The city experiences four distinct seasons, with warm summers, colorful autumns, cold winters, and mild springs.",
    "Bloomington has a vibrant arts culture, with annual festivals like the Lotus World Music & Arts Festival drawing international artists.",
    "The Kirkwood Avenue area near campus is a popular spot for students and locals, lined with cafes, bookstores, and music venues.",
]

# Initialize Chromadb and create the collection
client = chromadb.PersistentClient(path="mydb2/")
collection = client.get_or_create_collection(name="docs")
    

# Precompute document embeddings and add to the collection
for i, doc in enumerate(documents):
    response = ollama.embed(model="nomic-embed-text", input=doc)
    embeddings = response["embeddings"]
    collection.add(
        ids=[str(i)],
        embeddings=embeddings,
        documents=[doc]
    )

# Function to handle context retrieval and response generation
def get_relevant_context(prompt):
    # Generate embedding for the user's prompt
    prompt_response = ollama.embed(model="nomic-embed-text", input=prompt)
    prompt_embedding = prompt_response["embeddings"]
    results = collection.query(
        query_embeddings=prompt_embedding,
        n_results=1
    )
    relevant_document = results['documents'][0][0] if results and 'documents' in results else None
    return relevant_document

# Streamlit UI
st.title("Hope You FellBot App")
user_prompt = st.text_area("Enter a prompt to retrieve context:", height=200)

if st.button("Retrieve Context"):
    context = get_relevant_context(user_prompt)
    if context:
        st.subheader("Relevant Context:")
        st.write(context)
        st.subheader("Generated Response:")
        response = ollama.generate(
            model="hf.co/unsloth/Qwen3-8B-GGUF:UD-Q4_K_XL",
            prompt=f"Using this data: {context}. Respond to this prompt: {user_prompt}\n"
                  f"{user_prompt}")
        st.write(response.get('response', 'No response generated'))
    else:
        st.write("No relevant context found.")