# Large Language Models (LLMs)
Large Language Models (LLMs) are a cutting-edge technology within the field of artificial intelligence, specifically natural language processing (NLP). They are deep learning models trained on massive amounts of text data, enabling them to understand, generate, and manipulate human language with remarkable proficiency. LLMs are capable of performing a wide array of language-related tasks, including translation, summarization, question answering, and content creation. Their applications are vast and continuously expanding, ranging from powering chatbots and virtual assistants to aiding in research and analysis. 
For a more detailed description of the general structure of LLMs, we refer the interested reader to this excellent [post](https://jalammar.github.io/illustrated-transformer/)

This section describes how DeepProve proves the correct inference of a given LLM. The proving strategy assumes that the structure of the model (e.g., number and type of layers) is publicly known, while the actual weights and parameters of the model might be known only to the prover.

We first describe how to prove each of the major building blocks of LLMs. Then, we'll show how to prove an entire model composed by the major building blocks.

## LLMs Layers
- [QKV layer](llms-layers/qkv.md)
- [Positional Encoding](llms-layers/positionals.md)
- [MHA Layer](llms-layers/mha.md)
- [Embedding Layer](llms-layers/embeddings.md)
