#adapted from https://ollama.com/blog/embedding-models
#
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

client = chromadb.PersistentClient(path="mydb1/")
collection = client.create_collection(name="docs")

# store each document in a vector embedding database
for i, d in enumerate(documents):
  response = ollama.embed(model="nomic-embed-text", input=d)
  embeddings = response["embeddings"]
  collection.add(
    ids=[str(i)],
    embeddings=embeddings,
    documents=[d]
  )

# an example prompt
prompt = "What are healthy ways to handle conflict in a relationship??"

# generate an embedding for the input and retrieve the most relevant doc
response = ollama.embed(
  model="nomic-embed-text",
  input=prompt
)
results = collection.query(
  query_embeddings=response["embeddings"],
  n_results=1
)
data = results['documents'][0][0]

print("found relevant context:")
print(data)

# generate a response combining the prompt and data we retrieved
output = ollama.generate(
  model="hf.co/unsloth/Qwen3-8B-GGUF:UD-Q4_K_XL",
  prompt=f"Using this data: {data}. Respond to this prompt: {prompt}")

print(output['response'])
