from django import forms

class NLPInputForm(forms.Form):
    # Field 1: Text Area
    raw_text = forms.CharField(
        label="Enter Text",
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control', 
            'rows': 6, 
            'placeholder': 'Paste biomedical abstract here...'
        })
    )
    
    # Field 2: File Upload
    # We restrict extensions to .xml or .txt for safety
    xml_file = forms.FileField(
        label="Or Upload PubMed XML",
        required=False,
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control',
            'accept': '.xml, .txt' 
        })
    )

    def clean(self):
        """
        Validation logic: Ensure at least one field (text or file) is provided.
        """
        cleaned_data = super().clean()
        text = cleaned_data.get("raw_text")
        file = cleaned_data.get("xml_file")

        if not text and not file:
            raise forms.ValidationError("Please provide either raw text or an XML file.")
            
        return cleaned_data