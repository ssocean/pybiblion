import os
import json
import time
from urllib.error import URLError
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import arxiv
import threading

from cfg.safesession import session_factory
from database.DBEntity import PaperMapping
from retrievers.arxiv_paper import Arxiv_paper

from retry import retry
# 
@retry(tries=5)
def process_result(session, result, out_dir, key_word, query):
    key_words_count = {}
    error_list = []

    if '/' in result._get_default_filename() or '\\' in result._get_default_filename():
        return key_words_count, error_list  # 

    if result.published.year >= 2010:

        arxiv_paper = Arxiv_paper(result, ref_type='entity')
        data = session.query(PaperMapping).filter(PaperMapping.id == arxiv_paper.id).first()

        if data is None:  # 
            if not os.path.exists(os.path.join(out_dir, result._get_default_filename())):
                try:
                    print(result._get_default_filename()+'Downloading...')
                    result.download_pdf(dirpath=out_dir)
                except URLError:
                    time.sleep(4)
                    raise URLError('process_result error: result.download_pdf(dirpath=out_dir) Failed.')
                    error_list.append(result._get_default_filename())
                    return key_words_count, error_list

            doc = PaperMapping(arxiv_paper=arxiv_paper, search_by_keywords=query)
            doc.download_pth = result._get_default_filename()
            session.add(doc)
            session.commit()
        else:
            data.search_by_keywords = query
            session.commit()

        if key_word.lower() not in key_words_count:
            key_words_count[f'{key_word.lower()}'] = 1
        else:
            key_words_count[f'{key_word.lower()}'] += 1

    return key_words_count, error_list


def process_keyword(session, key_word, out_dir, lock, total_keywords_count, total_error_list):
    key_words_count = {}
    error_list = []

    # 
    query = f'(ti:"review" OR ti:"survey") AND (ti: "{key_word.lower()}" OR abs:"{key_word.lower()}")'

    search = arxiv.Search(query=query,
                          max_results=float('inf'),
                          sort_by=arxiv.SortCriterion.Relevance,
                          sort_order=arxiv.SortOrder.Descending)
    search_rst = []

    for result in search.results():
        search_rst.append(result)
    # 
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}

        # 
        for result in search_rst:
            future = executor.submit(process_result, session, result, out_dir, key_word, query)
            futures[future] = result

        for future in as_completed(futures):
            try:
                partial_keywords_count, partial_error_list = future.result()
                # 
                with lock:
                    for k, v in partial_keywords_count.items():
                        if k in total_keywords_count:
                            total_keywords_count[k] += v
                        else:
                            total_keywords_count[k] = v
                    total_error_list.extend(partial_error_list)
            except Exception as e:
                print(f"Error processing result for keyword '{key_word}': {str(e)}")


def main(session):
    key_words = ['Action Detection', 'Action Recognition', 'Activity Detection', 'Activity Recognition', 'Adversarial Attack',  'Anomaly Detection', 'Audio Classification', 'Biometric Authentication', 'Biometric Identification',  'Boundary Detection', 'CNN', 'Computer Vision', 'Contrastive Learning', 'Data Mining', 'Data Visualization',  'Depth Estimation', 'Dialogue Modeling', 'Dialogue Systems', 'Diffusion Model', 'Document Analysis',  'Document Analysis and Recognition', 'Document Clustering', 'Document Layout Analysis', 'Document Retrieval',  'Domain Adaptation', 'Edge Detection', 'Emotion Recognition', 'Face Detection', 'Face Recognition',  'Facial Recognition', 'Gesture Analysis', 'Gesture Recognition', 'Graph Mining', 'Hand Gesture Recognition',  'Handwriting Recognition', 'Human Activity Recognition', 'Human Detection', 'Human Pose Estimation',  'Image Captioning', 'Image Classification', 'Image Clustering', 'Image Compression', 'Image Editing',  'Image Enhancement', 'Image Generation', 'Image Inpainting', 'Image Matching', 'Image Quality Assessment',  'Image Recognition', 'Image Reconstruction', 'Image Restoration', 'Image Retrieval', 'Image Segmentation',  'Image-Based Localization', 'Instance Segmentation', 'Knowledge Graph', 'Knowledge Representation',  'Language Modeling', 'Language Modelling', 'Machine Learning Interpretability', 'Machine Translation',  'Medical Image Analysis', 'Medical Image Segmentation', 'Meta-Learning', 'Metric Learning',  'Multi-Label Classification', 'Named Entity Disambiguation', 'Named Entity Recognition',  'Natural Language Processing', 'Object Detection', 'Object Tracking', 'Optical Character Recognition',  'Pattern Matching', 'Pattern Recognition', 'Person Re-Identification', 'Point Cloud', 'Pre-training',  'Pretraining', 'Prompt Learning', 'Question Answering', 'Recommendation Systems', 'Recommender Systems',  'Relation Extraction', 'Remote Sensing', 'Representation Learning', 'Saliency Detection',  'Salient Object Detection', 'Scene Segmentation', 'Scene Understanding', 'Self-Supervised Learning',  'Semantic Segmentation', 'Sentiment Analysis', 'Sentiment Classification', 'Signature Verification',  'Speech Emotion Recognition', 'Speech Enhancement', 'Speech Recognition', 'Speech Synthesis',  'Speech-to-Text Conversion', 'Super-Resolution', 'Superpixels', 'Text Classification', 'Text Clustering',  'Text Generation', 'Text Mining', 'Text Summarization', 'Text-to-Image Generation', 'Text-to-Speech Conversion',  'Text-to-Speech Synthesis', 'Time Series Analysis', 'Time Series Forecasting', 'Topic Detection',  'Topic Modeling', 'Transfer Learning', 'Unsupervised Learning', 'Video Object Segmentation', 'Video Processing',  'Video Summarization', 'Video Understanding', 'Vision Language Model', 'Visual Question Answering',  'Visual Tracking', 'Word Embeddings', 'Zero-Shot Learning']


    key_words = list(set([i.lower() for i in key_words]))


    out_dir = r'J:\SLR'  # 
    total_keywords_count = {}
    total_error_list = []
    lock = threading.Lock()  # 

    # 
    for key_word in tqdm(key_words):
        process_keyword(session, key_word, out_dir, lock, total_keywords_count, total_error_list)

    session.close()

    # 
    with open(r"C:\Users\Ocean\Documents\GitHub\PyBiblion\tools\kwd_count.json", "w") as json_file:
        json.dump(total_keywords_count, json_file)

    for err in total_error_list:
        print(err)

if __name__ == '__main__':
    main(session_factory)

    # search = arxiv.Search(query=f'computer vision survey',
    #                           max_results=float('inf'),
    #                           sort_by=arxiv.SortCriterion.Relevance,
    #                           sort_order=arxiv.SortOrder.Descending)
    # p = []
    # for result in search.results():
    #     p.append(result)
    #
    # print(len(p))