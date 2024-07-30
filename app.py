from huggingface_hub import InferenceClient
import streamlit as st
import pandas as pd
from googletrans import Translator
import re
import string

# Function to convert each row in the dataframe
def convert(row):
    s = row['Pin of Interest']
    v = row[s]
    return f"Force {v} on {s} pin and measure the voltage on the same {s} pin with SPU with range of {row['Lower Limit']} and {row['Upper Limit']}."

# Function to clean text
def clean_text(text):
    printable_text = ''.join(char for char in text if char in string.printable)  # Remove non-printable characters
    cleaned_text = re.sub(r'[^\x00-\x7F]', '', printable_text)  # Remove non-ASCII characters
    return cleaned_text

# Function to translate text to English
def translate_to_english(text, src_lang='auto'):
    translator = Translator()
    translation = translator.translate(text, src=src_lang, dest='en')
    return translation.text

# Initialize Hugging Face clients
vishesh_client = InferenceClient("imvishesh007/gemma-Code-Instruct-Finetune-test",token="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")
madhavi_client = InferenceClient("imvishesh007/gemma-Code-Instruct-Finetune-test",token="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")
rakesh_client = InferenceClient("bandi333/gemma-Code-Instruct-Finetune-test-v0.0",token="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")
models = {
    "vishesh_client": vishesh_client,
    "rakesh_client": rakesh_client,      # Add your token and model for rakesh_client if needed
    "madhavi_client": madhavi_client      # Add your token and model for madhavi_client if needed
}
def process_client(client, df):
    x = ""
    for i in range(df.shape[0]):
        z = st.checkbox(df['english sentence'][i])
        if z:
            for message in client.chat_completion(messages=[{"role": "user", "content": df['english sentence'][i]}], max_tokens=500, stream=True):
                print(message.choices[0].delta.content, end="")
                x += message.choices[0].delta.content
    return x

def main():
    st.set_page_config(layout="wide", page_title="MODELS")

    # Sidebar UI for uploading file and selecting model
    st.sidebar.title("Model Selection and File Upload")
    uploaded_f = st.sidebar.file_uploader("Upload your Excel file", type=["csv"], key="testcase")
    selected_model = st.sidebar.selectbox("Choose your model", options=list(models.keys()))
    st.sidebar.title("Enhanced Dataset for Fine-Tuning: Upload ")
    uploader_f2 = st.sidebar.file_uploader("Upload your Excel file", type=["csv"],key="fine_tuning_dataset")
    
    if uploader_f2 is not None:
        
        from huggingface_hub import login
        login(token="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")
        from transformers import AutoModelForCausalLM, AutoTokenizer


        model_id = "bandi333/gemma-Code-Instruct-Finetune-test-v0.0"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        from datasets import Dataset
        import pandas as pd
        df = pd.read_csv(uploader_f2)
        dataset =  Dataset.from_pandas(df)
        def generate_prompt(data_point):
            """Gen. input text based on a prompt, task instruction, (context info.), and answer
        
            :param data_point: dict: Data point
            :return: dict: tokenzed prompt
            """
            prefix_text = 'Below is an instruction that describes a task. Write a response that ' \
                       'appropriately completes the request.\n\n'
            # Samples with additional context into.
            if data_point['input']:
                text = f"""<start_of_turn>user {prefix_text} {data_point["instruction"]} here are the inputs {data_point["input"]} <end_of_turn>\n<start_of_turn>model{data_point["output"]} <end_of_turn>"""
            # Without
            else:
                text = f"""<start_of_turn>user {prefix_text} {data_point["instruction"]} <end_of_turn>\n<start_of_turn>model{data_point["output"]} <end_of_turn>"""
            return text
        
        # add the "prompt" column in the dataset
        text_column = [generate_prompt(data_point) for data_point in dataset]
        dataset = dataset.add_column("prompt", text_column)
        dataset = dataset.shuffle(seed=1234)  # Shuffle dataset here
        dataset = dataset.map(lambda samples: tokenizer(samples["prompt"]), batched=True)
        dataset = dataset.train_test_split(test_size=0.2)
        train_data = dataset["train"]
        test_data = dataset["test"]
        from transformers import TrainingArguments, Trainer
        
        training_args = TrainingArguments(
            output_dir="./results",           
            evaluation_strategy="epoch",       
            learning_rate=2e-5,                 
            per_device_train_batch_size=4,      
            per_device_eval_batch_size=4,      
            num_train_epochs=3,                
            weight_decay=0.01,                 
            save_total_limit=3,                 
        )
        import transformers
        
        
        trainer = Trainer(
            model=model,                          
            train_dataset=train_data,
            eval_dataset=test_data,
             args=transformers.TrainingArguments(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                warmup_steps=0.03,
                max_steps=100,
                learning_rate=2e-4,
                logging_steps=1,
                output_dir="outputs",
                save_strategy="epoch",
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
                
            
        )
        
        
        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        
        model.save_pretrained("./fine-tuned-model")
        tokenizer.save_pretrained("./fine-tuned-model")
        from huggingface_hub import HfApi, Repository
        
        auth_token = "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"
        
        
        local_dir = "./fine-tuned-model"
        
        
        model_id = "bandi333/gemma-Code-Instruct-Finetune-test-v1.1"
        
        
        repo = Repository(
            local_dir=local_dir,         
            clone_from=model_id,          
            use_auth_token=auth_token      
        )
        
        repo.push_to_hub()

    if uploaded_f is not None:
        try:
            df = pd.read_csv(uploaded_f)

            # Display original dataframe
            st.subheader("Original Test Case File")
            st.dataframe(df)

            # Convert dataframe to English sentences
            df['english sentence'] = df.apply(convert, axis=1)

            # Display dataframe with English conversion
            st.subheader("Dataframe with English Conversion")
            st.dataframe(df['english sentence'])

            # Add prefix to English sentences
            promtg = "code for the given requirement using customlibrary in cpp for the pin configuration test case"
            df['english sentence'] = df['english sentence'].apply(lambda x: promtg + x)

            # Process selected model
            if selected_model in models and models[selected_model] is not None:
                st.subheader("Interact with Hugging Face Model")
                x = process_client(models[selected_model], df)
            else:
                st.warning(f"Model {selected_model} is not configured or available.")

            # Translate and clean the final output
            x = translate_to_english(x)
            x = clean_text(x)

            # Display final translated and cleaned output
            st.subheader("Final Translated and Cleaned Output")
            st.write(x)

        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

if __name__ == '__main__':
    main()
