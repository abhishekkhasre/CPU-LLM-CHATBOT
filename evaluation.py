from chat_chain import load_chain
import pandas as pd


chain = load_chain()

df  = pd.read_csv("./data/eval_questions.csv")

for idx, row in df.iterrows():
    print(f"Processing {idx}....")
    question = row["question"]
    reference_answer = row["reference_answer"]
    response = chain.invoke({"input": question})
    df.at[idx, "Bot_Answer"] = response["answer"]


with open("accuracy.csv" , "wb") as f:
    df.to_csv(f, index=False)