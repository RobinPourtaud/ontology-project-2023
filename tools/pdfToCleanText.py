def pdfToCleanText(pdfPath : str = "rawData/EU.pdf", outputPath : str = "data/corpus.txt", pages : list = None, returnCorpus : bool = True) -> str:
    """Convert pdf to clean text

    Args:
        pdfPath (str, optional): Path to pdf file. Defaults to "rawData/EU.pdf".
        outputPath (str, optional): Path to output file. Defaults to "data/EU.txt".
        pages (list, optional): List of pages to convert. Defaults to None.
        returnCorpus (bool, optional): Preprocess text. Defaults to True. It will return a set of sentences.

    Returns:
        str: Clean text from pdf
    """
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from pdfminer.pdfpage import PDFPage
    from io import StringIO

    if pages == None:
        pagenums = set(list(range(2, 59)) + list(range(61, 104)))
    else:
        pagenums = set(pages)

    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    infile = open(pdfPath, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close()

    if returnCorpus:
        # remove \n, \t, \r
        text = text.replace("\n", "").replace("\t", "").replace("\r", "").replace("\x0c", "").replace("\x0b", "").replace("15698/22 TREE.2.B RB/ek", "")
        # remove multiple spaces
        text = " ".join(text.split())

        from nltk.tokenize import sent_tokenize
        from nltk import download
        try:
            text = sent_tokenize(text)
        except Exception as e:
            download("punkt")
            text = sent_tokenize(text)
        # remove small sentences
        text = [sentence for sentence in text if len(sentence) > 10]
        # split sentences with spaces
        text = [sentence.split() for sentence in text]

        # keep only NP 
        
    else:
        text = text.split()

    with open(outputPath, "w") as f:
        for sentence in text:
            f.write(" ".join(sentence) + "\n")
    return text