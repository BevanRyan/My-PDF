# Langchain Ask PDF (Tutorial)

>You may find the step-by-step video tutorial to build this application [on Youtube](https://youtu.be/wUAUdEw5oxM).

This is a Python application that allows you to load a PDF and ask questions about it using natural language. The application uses a LLM to generate a response about your PDF. The LLM will not answer questions unrelated to the document.

## How it works

The application reads the PDF and splits the text into smaller chunks that can be then fed into a LLM. It uses OpenAI embeddings to create vector representations of the chunks. The application then finds the chunks that are semantically similar to the question that the user asked and feeds those chunks to the LLM to generate a response.

The application uses Streamlit to create the GUI and Langchain to deal with the LLM.


## Installation

To install the repository, please clone this repository and install the requirements:

```
pip install -r requirements.txt
```

You will also need to add your OpenAI API key to the `.env` file.

## Usage

To use the application, run the `main.py` file with the streamlit CLI (after having installed streamlit): 

```
streamlit run app.py
```


## Contributing

This repository is for educational purposes only and is not intended to receive further contributions. It is supposed to be used as support material for the YouTube tutorial that shows how to build the project.


## Optimizations
Parallelize Text Extraction: Currently, the code extracts text from each page of the PDF sequentially. You can speed up the process by using parallelization techniques, such as concurrent.futures or multiprocessing, to extract text from multiple pages simultaneously. This can significantly reduce the overall execution time.

Optimize Chunk Size: Experiment with different values for chunk_size and chunk_overlap in the CharacterTextSplitter constructor to find an optimal configuration. Larger chunk_size values may increase processing efficiency, while smaller chunk_overlap values may reduce redundant computations.

Preload PDF Reader: Instead of creating a new PdfReader object for each file upload, you can preload the PDF reader once and reuse it for subsequent uploads. This can eliminate the overhead of initializing the reader repeatedly.

Lazy Loading: Consider loading the language models, embeddings, and vector stores lazily, only when they are actually required. This can reduce startup time and memory usage.

Cache Similarity Search: Implement a caching mechanism to store the results of similarity searches. If a user asks the same question multiple times, you can retrieve the previously computed results from the cache instead of performing the search again. This can save processing time for repeated queries. Wactch this https://www.youtube.com/watch?v=lYDiSCDcxmc

Optimize I/O Operations: Depending on the size and frequency of PDF uploads, you may need to optimize I/O operations. For example, you can explore using a faster file storage system or optimize the way PDFs are read and processed.

Profile and Monitor: Use profiling tools to identify performance bottlenecks in your code. Monitor resource usage (CPU, memory, disk) to ensure efficient utilization and detect any potential issues.

