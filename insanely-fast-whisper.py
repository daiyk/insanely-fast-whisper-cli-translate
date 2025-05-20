#!/usr/bin/env python3
import click
import os
import time
# sample usage:
#   automatic-speech-recognition: python insanely-fast-whisper.py --device mps sample_file.mp4 [use openai/whisper-large-v3-turbo model]
#   text-translation: python insanely-fast-whisper.py --task translation --device mps sample_file.txt [use Helsinki-NLP/opus-mt-de-en model]
@click.command()
@click.option('--task', default='ASR', help='Pipeline to use. Default is "ASR"(automatic-speech-recognition). Other options include "translation".')
@click.option('--model', default='openai/whisper-large-v3-turbo', help='ASR model to use for speech recognition. Default is "openai/whisper-base" for ASR and "Helsinki-NLP/opus-mt-en-de" for translation. Model sizes include base, small, medium, large, large-v2. Additionally, try appending ".en" to model names for English-only applications (not available for large).')
@click.option('--device', default='cuda:0', help='Device to use for computation. Default is "cuda:0". If you want to use CPU, specify "cpu".')
@click.option('--dtype', default='float32', help='Data type for computation. Can be either "float32" or "float16". Default is "float32".')
@click.option('--batch-size', type=int, default=8, help='Batch size for processing. This is the number of audio files processed at once. Default is 8.')
@click.option('--chunk-length', type=int, default=30, help='Length of audio chunks to process at once, in seconds. Default is 30 seconds.')
@click.option('--translate', is_flag=True, help='Flag to enable translation. If set, the ASR mode will perform translation to English instead of transcription.')
@click.argument('audio_file', type=str)
def asr_cli(task, model, device, dtype, batch_size, chunk_length, translate, audio_file):
    from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
    import torch
    if task == "ASR":
        generate_config = {}
        if translate:
            generate_config['language'] = 'en'
            generate_config['task'] = 'translate'
        model_id = model
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        # Initialize the ASR pipeline
        # pipe = pipeline("automatic-speech-recognition",
        #                 model=model,
        #                 device=device,
        #                 torch_dtype=torch.float16 if dtype == "float16" else torch.float32,
        #                 generate_kwargs=generate_config)
        # better transformer has been deprecated in favor of torch.compile
        # if better_transformer:
        #     pipe.model = pipe.model.to_bettertransformer()

        # Perform ASR
        click.echo("Model loaded.")
        start_time = time.perf_counter()
        # Check if translation is enabled

        outputs = pipe(audio_file, chunk_length_s=chunk_length, batch_size=batch_size, return_timestamps=True)

        # Output the results
        click.echo(outputs)
        click.echo("Transcription complete.")
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        click.echo(f"ASR took {elapsed_time:.2f} seconds.")

        # Save ASR chunks to an SRT file
        audio_file_name = os.path.splitext(os.path.basename(audio_file))[0]
        srt_filename = f"{audio_file_name}.srt"
        with open(srt_filename, 'w', encoding="utf-8") as srt_file:
            prev = 0
            for index, chunk in enumerate(outputs['chunks']):
                prev, start_time = seconds_to_srt_time_format(prev, chunk['timestamp'][0])
                prev, end_time = seconds_to_srt_time_format(prev, chunk['timestamp'][1])
                srt_file.write(f"{index + 1}\n")
                srt_file.write(f"{start_time} --> {end_time}\n")
                srt_file.write(f"{chunk['text'].strip()}\n\n")
    ### Translation pipeline ###
    if task == "translation":
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        # Initialize the translation pipeline
        if model == "openai/whisper-large-v3-turbo":
            # fall back to the default model
            model_id = "Helsinki-NLP/opus-mt-de-en"
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        input_text = ""
        if not audio_file.endswith(('.wav','.mp3', '.flac', '.m4a', '.ogg')):
            try:
                with open(audio_file,'r', encoding='utf-8') as f:
                    input_text = f.read()
            except Exception as e:
                click.echo(f"Error reading audio file: {e}")
                return
        else:
            click.echo("Audio file is not a text file. Please provide a valid text file for translation.")
            return
        
        # Load tokenizer and model
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, 
                torch_dtype=torch_dtype, 
                low_cpu_mem_usage=True
            )
            model.to(device)
            pipe = pipeline(
                "translation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch_dtype,
            )
            click.echo("Model loaded, start translation task......")
            start_time = time.perf_counter()

            paragraphs = input_text.split('\n')
            translated_paragraphs = []
            for paragraph in paragraphs:
                if not paragraph.strip()== '':
                    translated_info = pipe(paragraph)
                    print(f"Translated raw output: {translated_info}")
                    translated_paragraph = translated_info[0]['translation_text']
                    if isinstance(translated_paragraph, str):
                        translated_paragraphs.append(translated_paragraph)
                    else:
                        translated_paragraphs.append(' ')
            translated_text = '\n'.join(translated_paragraphs)  

            # output the translated text
            click.echo("Translation complete.")
            click.echo(translated_text)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            click.echo(f"Translation took {elapsed_time:.2f} seconds.")

            # Save the translated text to a file
            output_file = os.path.splitext(audio_file)[0] + "_translated.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(translated_text)
            click.echo(f"Translated text saved to {output_file}.")
        except Exception as e:
            click.echo(f"Error during translation: {e}")
            return

def seconds_to_srt_time_format(prev, seconds):
    if not (isinstance(seconds, int) or isinstance(seconds, float)):
        seconds = prev
    else:
        prev = seconds
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    return (prev, f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}")

if __name__ == '__main__':
    asr_cli()
