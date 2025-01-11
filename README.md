# Brief About

This repo will seve as a way to track previous versions of the bare bones single-hop RAG. Each commit will correspond to a version of the RAG where a critical problem was faced, so I can use it as a teaching point and share how I overcame that roadblock.

For example, in this repo's first commit, there will be a version of the application in which the top k `retrieved contexts` will almost all be nearly identical for a given `question`. This will serve as a point to:
1. Showcase the problem
2. Discuss how it arised
3. Present and code the solution

## Notes On Stopping Points in the Video

1. Stop to show what the dataset looks like when you have the pipeline and everything set up. For example, show how there are tuples of (question, ground truth ans, ground truth context) by printing them out

2. Stop to show how under an old iteration, the code would have many of the nearly same retrieved contexts for a given question and that there was a flaw in the original process that needs to now be addressed –deduplication. The deduplication problem that needs to be addressed will be in the first code commit. Subsequent stopping points will correspond to separate commited versions of tha RAG system.

# Notes on process to incorporate into video

Notes on my code that will be used in the video.

## Informational flow

### FAISS Index and Context Embeddings

Overview of how we set up the vector store.

1. The embedding model encodes the contexts from the loaded data set. We'll call this `context_embeddings` (find deminsionality later).

2. We take the `dimension` along `context_embeddings.shape[1]` and define an `index` as `FAISS.IndexFlatL2(dimension)`.

3. The context embeddings are then added to the FAISS `index`.

### User Queries

When a user sends a query, the following occurs:

1. `generate_answer` is called and it retrieves the relevant `top_k` contexts (defaults to 5)

	- `generate_answer` calls `retrieve_contexts`
	- `retrieve_contexts` will then:
		1. encode the user query using the `SentenceTransformer.encode`
		2. Gather the corresponding `distances` and `indices` from an `index.search`
		3. It then gives all the retrieved contexts by mapping the indices to the **raw** `contexts` array.

2. A single `question` can have **many** relevant contexts and we generate an answer for each context. These answers are achieve by feeding a particular `context` and `question` into the `qa_pipeline`,

3. The answers are then sorted in descending order and we return a dict of the bes answer and all other answers.


### Recap

- context embeddings get mapped to an index in the FAISS index.
- we get relevant contexts for a given query by embedding the query
- do a FAISS index search with that embedded query to get indices
- use those indices as a lookup on the raw contexts.
- feed the raw contexts with a raw query into the embedding model's QA pipeline

