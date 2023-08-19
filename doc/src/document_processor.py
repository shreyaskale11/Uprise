


class DocumentChunkSplit:
    def __init__(self, document,chunk_size=512):
        self.document = document
        self.chunk_size = chunk_size

    # Split the document into chunks (adjust chunk size as needed)
    def split_document(self):
        chunks = [self.document[i:i+self.chunk_size] for i in range(0, len(self.document), self.chunk_size)]
        return chunks

class DocumentParaSplit:
    # Constructor
    def __init__(self, document):
        self.document = document

    # Split the document into sentences
    def split_into_sentences(self):
        sentences = []
        for sentence in self.document.split('\n'):
            sentences.append(sentence.strip())
        return sentences
    # Split the document into sentences
    def split_document(self):
        sentences = self.document.split('. ')
        return sentences
    # Split the document into paragraphs
    def split_into_paragraphs(self):
        paragraphs = []
        for paragraph in self.document.split('\n\n'):
            paragraphs.append(paragraph.strip())
        return paragraphs
    

# export this class to use in other modules
class DocumentTokenizer:
    # Constructor
    def __init__(self, document):
        self.document = document

    # Tokenize the document
    def tokenize(self):
        # Split the document into sentences
        sentences = DocumentParaSplit(self.document).split_into_sentences()
        # Tokenize each sentence
        tokenized_sentences = [sentence.split() for sentence in sentences]
        return tokenized_sentences

# export this class to use in other modules
class DocumentChunkTokenizer:
    # Constructor
    def __init__(self, document):
        self.document = document

    # Tokenize the document
    def tokenize(self):
        # Split the document into chunks
        chunks = DocumentChunkSplit(self.document).split_document()
        # Tokenize each chunk
        tokenized_chunks = [chunk.split() for chunk in chunks]
        return tokenized_chunks

