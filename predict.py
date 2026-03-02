from citron.citron import Citron
from citron.data import Quote
from citron import utils
from citron.data import DataSource
import json
from typing import List, NamedTuple, Tuple
import sys


def formatSentTokIdx(absTokOffsets, tok2sent):
    return [f"{tok2sent[t][0]}:{tok2sent[t][1]}" for t in absTokOffsets]

class AnnotationGroup(NamedTuple):
    msg: List[int]
    cue: List[int]
    addr: List[int]
    frame: List[int]
    src: List[int]
    form: str
    stwr: str

def writeJSON(
    output,
    name: str,
    sentences: List[List[str]],
    tokens: List[str],
    tok2sent: List[Tuple[int, int]],
    groups: List[AnnotationGroup],
):
    data = {"DocumentName": name}
    data["Sentences"] = [
        {"SentenceId": i, "Tokens": sent} for i, sent in enumerate(sentences)
    ]
    annos = []
    for msg, cue, addr, frame, src, form, stwr in groups:
        a = {
            "Addr": formatSentTokIdx(addr, tok2sent),
            "AddrText": " ".join(tokens[t] for t in addr),
            "Cue": formatSentTokIdx(cue, tok2sent),
            "CueText": " ".join(tokens[t] for t in cue),
            "Form": form,
            "Frame": formatSentTokIdx(frame, tok2sent),
            "FrameText": " ".join(tokens[t] for t in frame),
            "Message": formatSentTokIdx(msg, tok2sent),
            "MessageText": " ".join(tokens[t] for t in msg),
            "STWR": stwr,
            "Source": formatSentTokIdx(src, tok2sent),
            "SourceText": " ".join(tokens[t] for t in src),
        }
        annos.append(a)
    data["Annotations"] = annos
    json.dump(data, output, indent=2, ensure_ascii=False)

def chars2tokens(span, start2idx, end2idx, tokenStarts, tokenEnds):
    start, end = 0, len(tokenEnds) - 1
    if span.start_char in start2idx and span.end_char in end2idx:
        start = start2idx[span.start_char]
        end = end2idx[span.end_char] + 1
        
    else:
        print("tokenization is too different")
        for i, s in enumerate(tokenStarts):
            if s > span.start_char:
                start = i
                break
        for i, s in enumerate(tokenEnds):
            if s > span.end_char:
                end = i + 1
                break
    return range(start, end)

def spans2tokens(spans, start2idx, end2idx, tokenStarts, tokenEnds):
    return  [item for s in spans for item in chars2tokens(s, start2idx, end2idx, tokenStarts, tokenEnds)]

def toAnnoGroups(quotes: List[Quote], name: str, sentences: List[List[str]], tokens: List[str], tokenStarts: List[int], tokenEnds: List[int], tok2sent: List[Tuple[int,int]], sentenceOffsets):
    start2idx = {s:i for i,s in enumerate(tokenStarts)}
    end2idx = {s:i for i,s in enumerate(tokenEnds)}
    groups = []
    for q in quotes:
        msg = spans2tokens(q.contents, start2idx, end2idx, tokenStarts, tokenEnds)
        cue = chars2tokens(q.cue, start2idx, end2idx, tokenStarts, tokenEnds)
        src = spans2tokens(q.sources, start2idx, end2idx, tokenStarts, tokenEnds)
        sent = chars2tokens(q.cue.sent, start2idx, end2idx, tokenStarts, tokenEnds)
        frame = list(set(sent).difference(msg))
        frame.sort()
        # text = " ".join([tokens[t] for t in msg])
        #print(text)
        ag = AnnotationGroup(msg, cue, [], frame, src, "Direct" if q.contents[0].text[0] == "„" else "Indirect", "Speech")
        groups.append(ag)
    with open("outputs/test-cue1/" + name + ".json", "wt") as f:
        writeJSON(f, name, sentences, tokens, tok2sent, groups)

nlp = utils.get_parser()
citron = Citron("models/de_cue1", nlp)
for doc, actual_quotes, _, helper in DataSource(nlp, "/ltstorage/home/fschroeder/citron/test-cue1"):
    predicted_quotes = citron.get_quotes(doc, resolve_coreferences=False)
    toAnnoGroups(predicted_quotes, *helper)
    