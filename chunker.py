from haystack.components.preprocessors import DocumentSplitter

class Chunker:
    def __init__(self,text) -> None:
        self.text = text

    def chunk(self):
        page_splitter = DocumentSplitter(split_by='word', split_length=300, split_overlap=80)
        page_output = page_splitter.run(documents=self.text)
        isolated_pages = page_output["documents"]
        return isolated_pages
    







