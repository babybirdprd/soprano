from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from .vocos.decoder import SopranoDecoder
import torch
import re
from unidecode import unidecode
from scipy.io import wavfile
from huggingface_hub import hf_hub_download
import os


class SopranoTTS:
    def __init__(self,
            cache_size_mb=100,
            decoder_batch_size=None):
        cache_size_ratio = cache_size_mb * 1024**2 / torch.cuda.get_device_properties('cuda').total_memory
        print(cache_size_ratio)
        backend_config = TurbomindEngineConfig(cache_max_entry_count=cache_size_ratio)
        self.pipeline = pipeline('ekwek/Soprano-80M', # get ckpt path
            log_level='ERROR',
            backend_config=backend_config)
        self.decoder = SopranoDecoder().cuda()
        decoder_path = hf_hub_download(repo_id='ekwek/Soprano-80M', filename='decoder.pth')
        self.decoder.load_state_dict(torch.load(decoder_path)) # get ckpt path
        self.decoder_batch_size=decoder_batch_size
        if not decoder_batch_size:
            self.decoder_batch_size = 32
        self.STREAM_CHUNK = 4 # Decoder receptive field, do not change

        self.infer("Hello world!") # warmup

    def _preprocess_text(self, texts):
        '''
        adds prompt format and sentence/part index
        '''
        res = []
        for text_idx, text in enumerate(texts):
            text = text.strip()
            sentences = re.split(r"(?<=[.!?])\s+", text)
            processed_sentences = []
            for sentence_idx, sentence in enumerate(sentences):
                old_len = len(sentence)
                new_sentence = re.sub(r"[^A-Za-z !\$%&'*+,-./0123456789<>?_]", "", sentence)
                new_sentence = re.sub(r"[<>/_+]", "", new_sentence)
                new_sentence = re.sub(r"\.\.[^\.]", ".", new_sentence)
                new_len = len(new_sentence)
                if old_len != new_len:
                    print(f"Warning: unsupported characters found in sentence: {sentence}\n\tThese characters have been removed.")
                new_sentence = unidecode(new_sentence.strip())
                processed_sentences.append((f'[STOP][TEXT]{new_sentence}[START]', text_idx, sentence_idx))
            res.extend(processed_sentences)
        return res

    def infer(self,
            text,
            out_path=None,
            top_p=0.95,
            temperature=0.3,
            repetition_penalty=1.2):
        results = self.infer_batch([text],
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            out_dir=None)[0]
        if out_path:
            wavfile.write(f"{out_path}", 32000, results.cpu().numpy())
        return results

    def infer_batch(self,
            texts,
            out_dir=None,
            top_p=0.95,
            temperature=0.3,
            repetition_penalty=1.2):
        """
        split by sentence, infer, then batch decode
        """
        sentence_data = self._preprocess_text(texts)
        prompts = list(map(lambda x: x[0], sentence_data))
        gen_config=GenerationConfig(output_last_hidden_state='generation',
                            do_sample=True,
                            top_p=top_p,
                            temperature=temperature,
                            repetition_penalty=repetition_penalty,
                            max_new_tokens=512)
        responses = self.pipeline(prompts, gen_config=gen_config)
        hidden_states = []
        for i, response in enumerate(responses):
            if response.finish_reason != 'stop':
                print(f"Warning: some sentences did not complete generation, likely due to hallucination.")
            hidden_state = response.last_hidden_state
            hidden_states.append(hidden_state)
        combined = list(zip(hidden_states, sentence_data))
        combined.sort(key=lambda x: -x[0].size(0))
        hidden_states, sentence_data = zip(*combined)

        num_texts = len(texts)
        audio_concat = [[] for _ in range(num_texts)]
        for sentence in sentence_data:
            audio_concat[sentence[1]].append(None)
        for idx in range(0, len(hidden_states), self.decoder_batch_size):
            batch_hidden_states = []
            lengths = list(map(lambda x: x.size(0), hidden_states[idx:idx+self.decoder_batch_size]))
            N = len(lengths)
            for i in range(N):
                batch_hidden_states.append(torch.cat([
                    torch.zeros((1, 512, lengths[0]-lengths[i]), device='cuda'),
                    hidden_states[idx+i].unsqueeze(0).transpose(1,2).cuda().to(torch.float32),
                ], dim=2))
            batch_hidden_states = torch.cat(batch_hidden_states)
            with torch.no_grad():
                audio = self.decoder(batch_hidden_states)
            
            for i in range(N):
                text_id = sentence_data[idx+i][1]
                sentence_id = sentence_data[idx+i][2]
                audio_concat[text_id][sentence_id] = audio[i].squeeze()[-(lengths[i]*2048-2048):]
        audio_concat = [torch.cat(x) for x in audio_concat]
        
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            for i in range(len(audio_concat)):
                wavfile.write(f"{out_dir}/{i}.wav", 32000, audio_concat[i].cpu().numpy())
        return audio_concat

    def infer_stream(self,
            text,
            chunk_size=1,
            top_p=0.95,
            temperature=0.3,
            repetition_penalty=1.2):
        gen_config=GenerationConfig(output_last_hidden_state='generation',
                            do_sample=True,
                            top_p=top_p,
                            temperature=temperature,
                            repetition_penalty=repetition_penalty,
                            max_new_tokens=512)
        sentence_data = self._preprocess_text([text])

        for sentence, _, _ in sentence_data:
            responses = self.pipeline.stream_infer([sentence], gen_config=gen_config)
            hidden_states_buffer = []
            chunk_counter = chunk_size
            for token in responses:
                finished = token.finish_reason is not None
                if not finished: hidden_states_buffer.append(token.last_hidden_state[-1])
                hidden_states_buffer = hidden_states_buffer[-(2*self.STREAM_CHUNK+chunk_size):]
                if finished or len(hidden_states_buffer) >= self.STREAM_CHUNK + chunk_size:
                    if finished or chunk_counter == chunk_size:
                        batch_hidden_states = torch.stack(hidden_states_buffer)
                        inp = batch_hidden_states.unsqueeze(0).transpose(1, 2).cuda().to(torch.float32)
                        with torch.no_grad():
                            audio = self.decoder(inp)[0]
                        if finished:
                            audio_chunk = audio[-((self.STREAM_CHUNK+chunk_counter-1)*2048-2048):]
                        else:
                            audio_chunk = audio[-((self.STREAM_CHUNK+chunk_size)*2048-2048):-(self.STREAM_CHUNK*2048-2048)]
                        chunk_counter = 0
                        yield audio_chunk.cpu()
                    chunk_counter += 1
