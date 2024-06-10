import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
import re
from flair.splitter import SegtokSentenceSplitter


df = pd.read_csv("full.csv")

tagger = SequenceTagger.load("flair/ner-german-legal")
splitter = SegtokSentenceSplitter()

allowedOrgs = ["EU", "BGB", "Klarna", "PayPal", "Europäische Union", "Europäische Kommission", "AGB", "SOFORT", "Amazon",
               "Mastercard", "VISA", "American Express", "Klarnas", "easyCredit", "DHL", "Hermes", "Saferpay", "Sofortüberweisung.de",
               "Paypal", "Wirecard", "Giropay", "Payback", "Commerzbank", "Santander", "Consors", "EC", "V Pay", "Trusted Shops", "TÜV",
               "DSGVO", "Schufa","bank", "www"]

allowedUrls = ["europa.eu", "santander", "payback", "paypal", "klarna.com", "example.com", "easycredit", "trustedshops", "payments.amazon", "pay.amazon", "google.com"]

def isAllowed(text, allowed):
    for a in allowed:
        if a.lower() in text.lower():
            return True
    return False

cntOrg = 0
cntSt = 0
cntStr = 0

for ind in df.index:
    print(ind)
    if(ind < 2000):
        continue
    text = df['text'][ind]

    # regex

    ## remove email addresses
    pattern = "([\w\.\-\_]+@[\w\.\-\_]+)"
    s = re.search(pattern, text)
    if(s != None):
        #print(s.group(0))
        text = re.sub(pattern, 'hello@example.com', text)
        #df['text'][ind] = text

    ## remove urls
    pattern = "([-a-zA-Z0-9]{1,256}[.])?[-a-zA-Z0-9:%?._\\+~#=]{2,256}\\.[a-z]{2,5}"
    s = re.search(pattern, text)
    if (s != None and not isAllowed(s.group(0), allowedUrls)):
        #print(s.group(0))
        text = re.sub(pattern, 'example.com', text)
        #df['text'][ind] = text

    ## remove IBANs
    pattern = "[A-Z]{2}[0-9]{2}(?:[ ]?[0-9]{4}){4}(?!(?:[ ]?[0-9]){3})(?:[ ]?[0-9]{1,2})?"
    s = re.search(pattern, text)
    if (s != None):
        #print(s.group(0))
        text = re.sub(pattern, 'DE75512108001245126199', text)
        # df['text'][ind] = text

    ## remove IBANs
    pattern = "DE[0-9]{9}"
    s = re.search(pattern, text)
    if (s != None):
        #print(s.group(0))
        text = re.sub(pattern, 'DE398517849', text)
        # df['text'][ind] = text

    ## remove phone numbers
    pattern = "[(]?\+?[0-9]+([0-9]|\/|\(|\)|\-| ){9,}[0-9]"
    s = re.search(pattern, text)
    if (s != None):
        #print(s.group(0))
        text = re.sub(pattern, '00 00 12345678', text)
        # df['text'][ind] = text
        #cnt += 1

    ## remove zip codes
    pattern = "\b[0-9]{5}\b"
    s = re.search(pattern, text)
    if (s != None and not isAllowed(s.group(0), allowedUrls)):
        # print(s.group(0))
        text = re.sub(pattern, '00000', text)

    # ner
    sentences = splitter.split(text)
    tagger.predict(sentences)
    for sentence in sentences:
        for entity in sentence.get_spans('ner'):
            print(entity.tag)
            print(entity.text)
            # organisations
            if entity.tag in ["UN", "ORG", "PER"] and not isAllowed(entity.text, allowedOrgs):
                text = text.replace(entity.text, "<<NAME>>")
                cntOrg += 1
            elif entity.tag in ["ST"]:
                text = text.replace(entity.text, "<<STADT>>")
                cntSt += 1
            elif entity.tag in ["STR"]:
                text = text.replace(entity.text, "<<STRASSE>>")
                cntStr += 1

    df.loc[ind, "text"] = text

    if (ind%100) == 0:
        print("org: " + str(cntOrg))
        print("st: " + str(cntSt))
        print("str: " + str(cntStr))

        df.to_csv("../../corpus/anonym.csv", sep=',', index=False, encoding="utf-8")