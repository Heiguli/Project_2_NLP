#Get the precision, recal and f1 scores of reference and extracted keywords
def get_keyword_extraction_statistics(reference_keywords, extractor_keywords):

    reference_keywords = {
        key.strip().lower()
        for key in reference_keywords
        if key.strip()
    }

    extractor_keywords = {
        key.strip().lower()
        for key in extractor_keywords
        if key.strip()
    }
    total_correct = len(reference_keywords & extractor_keywords)

    precision = (total_correct / (total_correct + (len(extractor_keywords) - total_correct))) if extractor_keywords else 0.0
    recall = (total_correct / (total_correct + (len(reference_keywords) - total_correct))) if reference_keywords else 0.0
    f1 = 2 * ((precision * recall) / (precision + recall)) if precision + recall else 0.0
    return precision, recall, f1