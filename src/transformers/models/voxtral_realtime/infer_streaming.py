import asyncio

from transformers import (
    VoxtralRealtimeProcessor,
    VoxtralRealtimeForConditionalGeneration,
    TextIteratorStreamer,
)
from transformers.audio_utils import load_audio


model_id = "/raid/eustache/Voxtral-Mini-4B-Realtime-2602-hf"
processor = VoxtralRealtimeProcessor.from_pretrained(model_id)

audio = load_audio("/home/eustache_lebihan/add-voxstral-trmfs/macron.wav")
model = VoxtralRealtimeForConditionalGeneration.from_pretrained(model_id, device_map="cuda:0")


async def main():
    # 1. Create a queue and put the first audio chunk
    audio_queue = asyncio.Queue()
    audio_queue.put_nowait(audio[:processor.first_audio_chunk_num_samples])

    # 2. Call the processor once — input_features is an async generator backed by the queue
    inputs = processor(
        audio=audio_queue,
        is_streaming=True,
        return_tensors="pt",
        device=model.device,
        dtype=model.dtype,
    )

    # 3. Start generation in a background thread
    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    loop = asyncio.get_running_loop()
    async_gen = inputs.input_features

    # Bridge: wrap the async generator into a sync iterator so that
    # model.generate() (which runs in a thread) can consume it via the event loop.
    def sync_input_features():
        while True:
            future = asyncio.run_coroutine_threadsafe(async_gen.__anext__(), loop)
            try:
                yield future.result()
            except StopAsyncIteration:
                return

    gen_future = loop.run_in_executor(None, lambda: model.generate(
        input_ids=inputs.input_ids,
        num_delay_tokens=inputs["num_delay_tokens"],
        input_features=sync_input_features(),
        streamer=streamer,
    ))

    # 4. Push remaining audio chunks into the queue
    start_idx = processor.start_idx_second_audio_chunk
    while (end_idx := start_idx + processor.audio_chunk_num_samples) < audio.shape[0]:
        await audio_queue.put(audio[start_idx:end_idx])
        start_idx = end_idx - processor.feature_extractor.win_length // 2

    await audio_queue.put(None)  # signal end of audio

    # 5. Iterate over the streamer to get text chunks as they are generated
    print("Model output (streaming):", end=" ", flush=True)
    await loop.run_in_executor(None, lambda: [print(t, end="", flush=True) for t in streamer])

    await gen_future


asyncio.run(main())
