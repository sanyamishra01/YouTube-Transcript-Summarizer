from youtube_transcript_api import YouTubeTranscriptApi
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import T5Tokenizer, T5ForConditionalGeneration


#Retriving data
video_id = "h5gNSHcoVmQ"
transcript = YouTubeTranscriptApi.get_transcript(video_id)

#Preprocessing data
text = ''.join([t['text'] for t in transcript])
clean_text = ''.join(e for e in text if e.isalnum() or e.isspace())
sentences = sent_tokenize(clean_text)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
stemmed_sentences = [''.join(stemmer.stem(word) for word in sentence.split() if word.lower() not in stop_words) for sentence in sentences]

#using Tf-Idf Vectorizer
tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length = 1024)
model = T5ForConditionalGeneration.from_pretrained('t5-base')

#summarize
inputs = tokenizer.encode("summarize: " + clean_text, return_tensors = 'pt', truncation = True)
summary_ids = model.generate(inputs, num_beams = 4, min_length = 500, max_length = 1000, length_penalty = 2.0)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens = True)
print(summary)