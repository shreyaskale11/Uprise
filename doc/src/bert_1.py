from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from document_processor import DocumentChunkSplit
# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# Define your large document
document = """
sophisticated and effective manner. They are
widely used across diverse fields such as scientific research papers, medical reports, and financial reports. Among these, financial reports are a
quintessential example of hybrid documents, distinguished by their intricate structure and the sheer
volume of numerical data interspersed within text
and tables. They play a crucial role in providing comprehensive insights for businesses and investors (Brito et al., 2019; Krause and Arora, 2008),
making them particularly noteworthy among various types of hybrid documents.
In order to investigate the potential of LLMs in
extracting and understanding information from hybrid documents, we design a task focused on Key
Performance Indicator (KPI)-related information
extraction from financial reports. Existing methods
attempting to address this challenge face several
limitations, such as inflexible rule-based named entity recognition methods, a focus on structured data,
and input token constraints that hinder comprehensive examination of financial reports (Farmakiotou
et al., 2000a; Brito et al., 2019; Chen et al., 2022;
Zhu et al., 2021; Zhao et al., 2022).
To address the challenges in extracting KPIs
from financial reports, we propose the Automatic
Financial Information Extraction (AFIE) framework, a comprehensive approach that leverages
LLMs. The AFIE framework comprises four main
modules—Segmentation, Retrieval, Summarization, and Extraction—working in synergy to efficiently extract keyword-corresponding values from
financial reports. With its carefully designed modules, AFIE exhibits three key features: 1) efficiently
handling long documents, 2) resolving ambiguity
in keyword representation, and 3) demonstrating
sensitivity to numerical values, effectively tackling
the challenges of KPI-related information extraction from financial reports. Our contributions can
be summarized as follows:
1. We propose a comprehensive framework for
arXiv:2305.16344v1 [cs.CL] 24 May 2023
Automatic Financial Information Extraction
(AFIE) designed to extract KPIs from financial reports. The AFIE framework comprises
four main modules—Segmentation, Retrieval,
Summarization, and Extraction—aimed at efficiently extracting keyword-corresponding
values from financial reports using LLMs.
2. To evaluate the extraction accuracy of LLMs,
we introduce the Financial Reports Numerical
Extraction (FINE) dataset. Rigorous quality
control measures have been implemented to
ensure the accuracy and relevance of each individual example. The FINE dataset can be
utilized to assess the performance of LLMs
in extracting information from financial documents.
"""


docsplit = DocumentChunkSplit(document)
chunks = docsplit.split_document()


# Define your question
question = "What is the main topic of this document?"

# Initialize an empty list to store answers
answers = []

# Process each chunk
for chunk in chunks:
    # Tokenize and encode the question and chunk
    inputs = tokenizer(question, chunk, return_tensors="pt")

    # Get the model's output
    outputs = model(**inputs)
    start_scores, end_scores = outputs.start_logits, outputs.end_logits

    # Find the answer span with the highest scores
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1

    # Decode the answer span
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

    # Append the answer to the list
    answers.append(answer)

# Combine answers from all chunks
combined_answer = " ".join(answers)

# Print the combined answer
print("Combined Answer:", combined_answer)
