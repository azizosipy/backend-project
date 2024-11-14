# nlp_app/views.py
import os
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Documents
from core.analyser.pdf_classification import extract_text_from_pdf, cosine_similarity_analysis  # Import your functions

OUTPUT_DIR = "extracted_files"

@api_view(['POST'])
def process_pdf(request):
    # Check if the file is provided in the request
    if 'file' not in request.FILES:
        return Response({"error": "No file uploaded."}, status=status.HTTP_400_BAD_REQUEST)
    
    # Save the uploaded file to the Documents model
    uploaded_file = request.FILES['file']
    document = Documents(file=uploaded_file)
    document.save()

    # Save the file to the directory for processing
    file_path = os.path.join(OUTPUT_DIR, uploaded_file.name)
    with open(file_path, 'wb') as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)
    
    # Extract text from the uploaded PDF
    extract_text_from_pdf(file_path)

    # Define the path to the sexual words file
    sexual_words_file = os.path.join(OUTPUT_DIR, r"C:\Users\oussa\OneDrive\Desktop\annabi\backend-project\pdf_classification\sexual_words.txt")  # Adjust path if necessary
    
    # Perform the cosine similarity analysis to detect sexual words
    sexual_words_found = cosine_similarity_analysis(file_path, OUTPUT_DIR, sexual_words_file)
    
    # Return the sexual words found as a response
    return Response({"sexual_words_found": sexual_words_found}, status=status.HTTP_200_OK)
