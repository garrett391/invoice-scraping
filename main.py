from transformers import pipeline
from PIL import Image
import os

# Set up paths relative to current directory
current_dir = os.getcwd()
invoice_dir = os.path.join(current_dir, "data", "invoice-images")
output_dir = os.path.join(current_dir, "data", "processed")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the image and convert to RGBA to avoid the transparency issue
image_filename = "invoice-template-overview.webp"
image_path = os.path.join(invoice_dir, image_filename)
image = Image.open(image_path).convert("RGBA")

# Save converted image in processed folder
output_filename = os.path.splitext(image_filename)[0] + "_rgba.png"
output_path = os.path.join(output_dir, output_filename)
image.save(output_path)  # Save the converted image

# Load the converted image in the pipeline
nlp = pipeline(
    "document-question-answering",
#    model="impira/layoutlm-document-qa",
#    model = "impira/layoutlm-invoices"
     model = "naver-clova-ix/donut-base-finetuned-docvqa"
)

# Get answers to each question and store the result
invoice_number = nlp(output_path, "What is the No number on the invoice?")
result_purchase_amount = nlp(output_path, "What is the total of the bill?")
result_billed_to = nlp(output_path, "Who is the invoice billed to?")
mail = nlp(output_path, "what is the email of billed to customer ?")

# Print the results
print("What is the invoice number:", invoice_number[0])
print("Purchase Amount:", result_purchase_amount[0])
print("Billed to:", result_billed_to[0])
print("Email is:", mail[0])