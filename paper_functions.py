import nltk

def nest_sentences(document,chunk_length):
    '''
    function to chunk a document
    input:  document           - Input document
            chunk_length        - chunk length
    output: list of chunks. Each chunk is a string.
    '''
    nested = []
    sent = []
    length = 0
    for sentence in nltk.sent_tokenize(document):
        length += len(sentence.split(" "))
        if length < chunk_length:
            sent.append(sentence)
        else:
            nested.append(" ".join(sent))
            sent = []
            sent.append(sentence)
            length = 0
    if len(sent)>0:
        nested.append(" ".join(sent))
    return nested