class PipelineConfig:
    def __init__(self, chunking="paragraph", embedding="all-mpnet-base-v2"):
        self.chunking = chunking
        self.embedding = embedding