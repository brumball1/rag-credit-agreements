class PipelineConfig:
    def __init__(self, chunking="page", embedding="mpnet", agentic=False):
        self.chunking = chunking
        self.embedding = embedding
        self.agentic = agentic