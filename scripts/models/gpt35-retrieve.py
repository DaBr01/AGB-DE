from openai import OpenAI
import pandas as pd
import json


client = OpenAI(api_key="xxx")

df = pd.read_csv("../../corpus/agb-de-anonym.csv")

out = {'responses': []}

for ind in df.index:
  mid = df["id"][ind]

  mtext = df["text"][ind]

  print(str(mid) + ": " + mtext)

  response = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    response_format= { "type": "json_object" },
    messages=[
      {"role": "system", "content": 'Stell dir vor, du bist ein Anwalt für Verbraucherschutz und berätst Verbraucher. Ist folgende Klausel in den AGB eines Online Shops potenziell ungültig wenn es sich beim Kunden um einen Verbraucher und beim Anbieter um eine Unternehmen handelt? Antworte mit true, wenn die Klausel potenziell ungültig ist und mit false wenn sie wahrscheinlich nicht ungültig ist. Erkläre deine Entscheidung. Antworte in folgendem JSON Format {"id": ' + str(mid) + ', "ungültig": Boolean, "erklärung": String}'},
      {"role": "user", "content": mtext}
    ]
  )

  js = json.loads(response.choices[0].message.content)
  out['responses'].append(js)

  with open("../../responses/gpt35_responses.json", "w", encoding="utf-8") as file:
    json.dump(out, file, indent=4, ensure_ascii=False)