from turbo import ChatterboxTurboTTS

model = ChatterboxTurboTTS.from_pretrained(device="cuda:0")

text = "Oh, that's hilarious! [laugh] Um anyway, do we have a new model in store?"

chunks = []
for chunk in model.generate_stream(
    text,
    chunk_size=10,  # 10 tokens = ~400ms per chunk
    overlap_tokens=5,  # 2 tokens overlap for continuity
):
    chunks.append(chunks)
