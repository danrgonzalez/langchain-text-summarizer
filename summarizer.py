from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Initialize the LLM
llm = OpenAI(temperature=0)

# Create a prompt template for summarization
prompt_template = """Write a concise summary of the following text:
"{text}"
CONCISE SUMMARY:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Create the LLMChain for summarization
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Create the StuffDocumentsChain
stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="text"
)

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len
)

def summarize_text(input_text: str) -> str:
    """
    Process and summarize the input text using LangChain components.
    
    Args:
        input_text (str): The text to be summarized
        
    Returns:
        str: The generated summary
    """
    # Split the text into chunks if necessary
    docs = text_splitter.create_documents([input_text])
    
    # Process the documents through the chain
    summary = stuff_chain.run(docs)
    
    return summary

# Example usage
if __name__ == "__main__":
    sample_text = """
    The Industrial Revolution marked a major turning point in Earth's ecology and humans' relationship with their environment. The Industrial Revolution dramatically changed every aspect of human life and lifestyles. The impact on the world's psyche would not begin to register until the early 1960s, some 200 years after its beginnings.
    
    From human development, health, and life longevity, to social improvements and the impact on natural resources, public health, energy usage and sanitation, the effects were profound. Improvements in technology and manufacturing processes led to the ability to produce more consumer goods at lower costs, leading to increased production and consumption.
    """
    
    summary = summarize_text(sample_text)
    print("Generated Summary:")
    print(summary)