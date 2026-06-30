from haystack.components.converters import PyPDFToDocument
from pathlib import Path

class Loader:
    def __init__(self,file_path) -> None:
        self.filepath = file_path


    def load(self):
        converter = PyPDFToDocument()
        output = converter.run(sources=[Path(self.filepath)])
        documents = output["documents"]
        #print(documents)
        return documents