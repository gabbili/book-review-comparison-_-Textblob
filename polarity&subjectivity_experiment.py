import os
import pandas as pd
from docx import Document
from pandas.core.window import doc
from textblob import TextBlob

def read_docx(filepath):
    doc = Document(filepath)
    full_text = [para.text for para in doc.paragraphs if para.text.strip()]
    return '\n'.join(full_text)

def analyze_text(text):
    blob = TextBlob(text)
    overall_polarity = blob.sentiment.polarity
    overall_subjectivity = blob.sentiment.subjectivity
    sentences_data = []
    for sentence in blob.sentences:
        sentences_data.append({
            'sentence': str(sentence),
            'polarity': sentence.sentiment.polarity,
            'subjectivity': sentence.sentiment.subjectivity
        })
    return overall_polarity, overall_subjectivity, sentences_data

def classify_by_bins(sentences_data, polarity_bins, subjectivity_bins):
    polarities = [s['polarity'] for s in sentences_data]
    subjectivities = [s['subjectivity'] for s in sentences_data]
    total = len(sentences_data)

    pol_counts = pd.cut(polarities, bins=polarity_bins, include_lowest=True).value_counts().sort_index()
    sub_counts = pd.cut(subjectivities, bins=subjectivity_bins, include_lowest=True).value_counts().sort_index()

    pol_percentages = (pol_counts / total *100).round(2)
    sub_percentages = (sub_counts / total *100).round(2)

    pol_labels = [f"{b:.2f}-{polarity_bins[i+1]:.2f}"for i, b in enumerate(polarity_bins[:-1])]
    sub_labels = [f"{b:.2f}-{subjectivity_bins[i+1]:.2f}"for i, b in enumerate(subjectivity_bins[:-1])]

    pol_dist = pd.DataFrame({
        'range': pol_labels,
        'count': pol_counts.values,
        'frequency': pol_percentages.values
    })
    sub_dist = pd.DataFrame({
        'range': sub_labels,
        'count': sub_counts.values,
        'frequency': sub_percentages.values
    })
    return pol_dist, sub_dist

def process_folder(folder_path, polarity_bins, subjectivity_bins):
    files = [f for f in os.listdir(folder_path) if f.endswith('.docx')]
    if not files:
        print(f"No docx files found in {folder_path}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0

    doc_summaries = []
    all_sentences = []
    combined_text = ''

    for file in files:
        file_path = os.path.join(folder_path, file)
        text = read_docx(file_path)
        combined_text += text +'\n'

        overall_polarity, overall_subjectivity, sentences_data = analyze_text(text)
        doc_summaries.append({
            'name': file,
            'overall_polarity': overall_polarity,
            'overall_subjectivity': overall_subjectivity,
        })
        all_sentences.extend(sentences_data)

    overall_polarity_comb, overall_subjectivity_comb, _ = analyze_text(combined_text)
    doc_summaries.append({
        'name': 'combined_text',
        'overall_polarity': overall_polarity_comb,
        'overall_subjectivity': overall_subjectivity_comb,
    })

    doc_df = pd.DataFrame(doc_summaries)

    pol_dist, sub_dist = classify_by_bins(all_sentences, polarity_bins, subjectivity_bins)
    return doc_df, pol_dist, sub_dist, len(all_sentences)

def main():
    human_folder = '/Users/jiechenli/Desktop/book review/experiment/human'
    machine_folder = '/Users/jiechenli/Desktop/book review/experiment/GPT'
    output_excel = 'sentiment_analysis_results.xlsx'

    polarity_bins = [-1.0, -0.5, 0.0, 0.5, 1.0]
    subjectivity_bins = [0.0, 0.25, 0.5, 0.75, 1.0]

    print('processing human reviews')
    human_doc, human_pol, human_sub, human_sent_cnt = process_folder(human_folder, polarity_bins, subjectivity_bins)
    print('processing machine reviews')
    machine_doc, machine_pol, machine_sub, machine_sent_cnt = process_folder(machine_folder, polarity_bins, subjectivity_bins)

    with pd.ExcelWriter(output_excel) as writer:
        if not human_doc.empty:
            human_doc.to_excel(writer, sheet_name='Human polarities', index=False)
            human_pol.to_excel(writer, sheet_name='Human polarity distribution', index=False)
            human_sub.to_excel(writer, sheet_name='Human subjectivity distribution', index=False)
        if not machine_doc.empty:
            machine_doc.to_excel(writer, sheet_name='Machine polarities', index=False)
            machine_pol.to_excel(writer, sheet_name='Machine polarity distribution', index=False)
            machine_sub.to_excel(writer, sheet_name='Machine subjectivity distribution', index=False)

    abs_path = os.path.abspath(output_excel)
    print(f'The analysis is finished, and the results are saved to {abs_path}')
    print(f'The human reviews counts: {human_sent_cnt}')
    print(f'The machine reviews counts: {machine_sent_cnt}')

if __name__ == '__main__':
    main()