# Brief About

This repo will seve as a way to track previous versions of the bare bones single-hop RAG. Each commit will correspond to a version of the RAG where a critical problem was faced, so I can use it as a teaching point and share how I overcame that roadblock.

For example, in this repo's first commit, there will be a version of the application in which the top k `retrieved contexts` will almost all be nearly identical for a given `question`. This will serve as a point to:
1. Showcase the problem
2. Discuss how it arised
3. Present and code the solution

### Notes On Stopping Points in the Video

1. Stop to show what the dataset looks like when you have the pipeline and everything set up. For example, show how there are tuples of (question, ground truth ans, ground truth context) by printing them out

2. Stop to show how under an old iteration, the code would have many of the nearly same retrieved contexts for a given question and that there was a flaw in the original process that needs to now be addressed –deduplication. The deduplication problem that needs to be addressed will be in the first code commit. Subsequent stopping points will correspond to separate commited versions of tha RAG system.
