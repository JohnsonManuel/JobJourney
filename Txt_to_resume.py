from fpdf import FPDF

def text_to_pdf(text, output_pdf_path):
    # Create instance of FPDF class
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Set font: Arial, bold, 12pt
    pdf.set_font("Arial", size=12)

    # Sanitize text by encoding in ascii and ignoring unsupported characters
    sanitized_text = text.encode('ascii', 'ignore').decode()

    # Add the sanitized text to the PDF
    pdf.multi_cell(0, 10, sanitized_text)

    # Output the PDF
    pdf.output(output_pdf_path)

# Example Usage
generated_text = """
Johnson Manuel R
Email: johnsonmanuel321@gmail.com
Phone: +49 1712793043
Location: Cottbus, Germany

PROFESSIONAL EXPERIENCE
Tata Consultancy Services | Systems Engineer | Chennai, India | Jan 2020 – Jan 2023
- Evaluated customer requirements and implemented change requests.
- Developed and maintained Java-based functionality for the application.
...
"""
output_pdf_path = "generated_resume.pdf"
text_to_pdf(generated_text, output_pdf_path)
