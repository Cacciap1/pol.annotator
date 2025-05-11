import openai
import pandas as pd
import time
from openai.error import RateLimitError
import tiktoken

#AI supported 
#ChatGPT3.5
#ChatGPT4

# Load the dataset
input_file = 'Insert dataset directory'
data = pd.read_excel(input_file)

# Define your OpenAI API key
openai.api_key = 'Insert your API KEY'

# Load the codebook content
with open('Insert your directory for codebook', 'r') as file:
    codebook_content = file.read()

# Extract definitions and examples from the codebook
codebook_parts = codebook_content.split("Text:")
variable_definitions = codebook_parts[0].strip()  # Everything before the first example is definitions
examples = "Text:" + "Text:".join(codebook_parts[1:]).strip()  # All annotated examples

# Function to calculate token count using tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(text):
    return len(encoding.encode(text))

# Calculate token counts for the dataset
data['token_count'] = data['text'].apply(lambda x: count_tokens(x) if pd.notna(x) else 0)

# Calculate total tokens, average tokens per row, and estimate costs
total_tokens = data['token_count'].sum()
average_tokens_per_row = data['token_count'].mean()
cost_per_1k_tokens = 0.002  # GPT-3.5-turbo pricing
estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens

# Estimate time (assuming 2 seconds per annotation)
time_per_annotation = 2  # in seconds
total_time = len(data) * time_per_annotation  # in seconds
total_time_minutes = total_time / 60

# Display estimates to the user
print(f"Dataset contains {len(data)} rows.")
print(f"Average tokens per row: {average_tokens_per_row:.2f}")
print(f"Total tokens: {total_tokens}")
print(f"Estimated cost: ${estimated_cost:.2f}")
print(f"Estimated time: {total_time_minutes:.2f} minutes")

# Ask user for confirmation to proceed
user_confirmation = input("Do you want to proceed with the annotation? (yes/no): ").strip().lower()
if user_confirmation != 'yes':
    print("Annotation process aborted by the user.")
    exit()

# Function to call ChatGPT API for annotation with backoff
def annotate_with_gpt(messages, retry_attempts=5):
    attempt = 0
    while attempt < retry_attempts:
        try:
            # Send the request to OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            # Extract the response text
            label = response['choices'][0]['message']['content'].strip()
            return label
        except RateLimitError as e:
            attempt += 1
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Rate limit hit. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Error processing text:\nError: {e}")
            return "error"
    return "rate_limit_error"

# Add a new column for GPT-based annotations
data['gpt_annotation'] = None

# Process dataset in chunks (batches) for large datasets
batch_size = 50  # Define the number of rows per batch

for start in range(0, len(data), batch_size):
    end = start + batch_size
    batch = data.iloc[start:end]

    # Initialize conversation with codebook for the batch
    messages = [
        {"role": "system", "content": "You are an assistant tasked with labeling text."},
        {"role": "user", "content": f"Here are the definitions of the variables:\n{variable_definitions}\n\nHere are examples to guide you:\n{examples}"}
    ]

    for idx, row in batch.iterrows():
        text = row['text']
        if pd.isna(text):
            data.at[idx, 'gpt_annotation'] = 'not_applicable'
            continue

        # Add text to the conversation
        messages.append({"role": "user", "content": f"Label the following text:\n{text}"})

        # Annotate the text
        label = annotate_with_gpt(messages)
        data.at[idx, 'gpt_annotation'] = label

        # Remove the text from the conversation to save context space
        messages.pop()

    # Save progress after each batch
    output_file = f'C:\\Users\\LENOVO\\Desktop\\Nuova cartella\\results\\Batch_{start}_{end}.xlsx'
    batch.to_excel(output_file, index=False)
    print(f"Batch {start}-{end} saved in {output_file}")

# Save the complete annotated dataset
output_file = 'Insert here the output directory'
data.to_excel(output_file, index=False)
print(f"Complete annotated dataset saved to {output_file}")
