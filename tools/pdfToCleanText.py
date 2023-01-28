def pdfToCleanText(pdfPath : str = "rawData/EU.pdf", outputPath : str = "data/EU.txt", pages : list = None) -> str:
    """Convert pdf to clean text

    Args:
        pdfPath (str, optional): Path to pdf file. Defaults to "rawData/EU.pdf".
        outputPath (str, optional): Path to output file. Defaults to "data/EU.txt".

    Returns:
        str: Clean text from pdf
    """
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from pdfminer.pdfpage import PDFPage
    from io import StringIO

    if not pages:
        pagenums = set()
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

    with open(outputPath, "w") as f:
        f.write(text)
    return ' '.join(text.split())