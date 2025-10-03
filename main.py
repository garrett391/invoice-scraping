from transformers import pipeline
from PIL import Image
import os
import pytesseract
import pandas as pd

# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

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
    model = "impira/layoutlm-invoices"
#     model = "naver-clova-ix/donut-base-finetuned-docvqa"
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


# Create a DataFrame with the results
results_df = pd.DataFrame({
    'Image_File': [image_filename],
    'Invoice_Number': [invoice_number[0]['answer']],
    'Invoice_Number_Score': [invoice_number[0]['score']],
    'Purchase_Amount': [result_purchase_amount[0]['answer']],
    'Purchase_Amount_Score': [result_purchase_amount[0]['score']],
    'Billed_To': [result_billed_to[0]['answer']],
    'Billed_To_Score': [result_billed_to[0]['score']],
    'Email': [mail[0]['answer']],
    'Email_Score': [mail[0]['score']]
})

# Save to CSV
csv_path = os.path.join(output_dir, "invoice_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")

# Display the table
print("\nResults Table:")
print(results_df.to_string(index=False))