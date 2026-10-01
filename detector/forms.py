from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError
import os


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = single_file_clean(data, initial)
        return result


class DocumentUploadForm(forms.Form):
    """
    Form to upload one or more requirement documents (.pdf, .docx, .txt)
    and configure detection parameters.
    """
    title = forms.CharField(
        max_length=255,
        required=False,
        initial="Requirements Conflict Analysis",
        widget=forms.TextInput(attrs={
            'class': 'form-control form-control-lg',
            'placeholder': 'e.g. Healthcare SRS v1 vs Billing BRD v2'
        })
    )

    documents = MultipleFileField(
        required=True,
        widget=MultipleFileInput(attrs={
            'class': 'form-control',
            'id': 'documentUploadInput',
            'accept': '.pdf,.docx,.txt'
        }),
        help_text="Upload one or more PDF, DOCX, or TXT requirement documents (Max 15MB each)."
    )

    similarity_threshold = forms.FloatField(
        required=False,
        initial=0.50,
        min_value=0.20,
        max_value=0.95,
        widget=forms.NumberInput(attrs={
            'class': 'form-range',
            'id': 'thresholdRange',
            'type': 'range',
            'step': '0.05',
            'min': '0.20',
            'max': '0.95'
        }),
        help_text="Cosine similarity threshold for selecting candidate pairs to be evaluated by the LLM."
    )

    cross_document_only = forms.BooleanField(
        required=False,
        initial=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input',
            'id': 'crossDocCheck'
        }),
        help_text="Check to only compare requirements across different documents (ignore internal intra-document pairs)."
    )

    def clean_documents(self):
        uploaded_files = self.files.getlist('documents')
        if not uploaded_files:
            raise ValidationError("Please select at least one document to analyze.")

        allowed_extensions = getattr(settings, 'ALLOWED_EXTENSIONS', ['.pdf', '.docx', '.txt'])
        max_size_mb = getattr(settings, 'MAX_UPLOAD_SIZE_MB', 15)
        max_size_bytes = max_size_mb * 1024 * 1024

        for file_obj in uploaded_files:
            name = file_obj.name.lower()
            ext = os.path.splitext(name)[1]
            if ext not in allowed_extensions:
                raise ValidationError(
                    f"File '{file_obj.name}' has unsupported format '{ext}'. Allowed: {', '.join(allowed_extensions)}"
                )

            if file_obj.size > max_size_bytes:
                raise ValidationError(
                    f"File '{file_obj.name}' exceeds the maximum allowed size of {max_size_mb}MB."
                )

        return uploaded_files


class UserRegisterForm(forms.Form):
    """
    User registration form with email, username, and password confirmation.
    """
    username = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={
            'class': 'form-control form-control-lg',
            'placeholder': 'Enter your username',
            'autocomplete': 'username',
        })
    )
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control form-control-lg',
            'placeholder': 'Enter your email address',
            'autocomplete': 'email',
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control form-control-lg',
            'placeholder': 'Choose a strong password',
            'autocomplete': 'new-password',
        })
    )
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control form-control-lg',
            'placeholder': 'Confirm your password',
            'autocomplete': 'new-password',
        })
    )

    def clean_username(self):
        from django.contrib.auth.models import User
        username = self.cleaned_data.get('username')
        if User.objects.filter(username__iexact=username).exists():
            raise ValidationError("This username is already taken. Please choose another.")
        return username

    def clean_email(self):
        from django.contrib.auth.models import User
        email = self.cleaned_data.get('email')
        if User.objects.filter(email__iexact=email).exists():
            raise ValidationError("An account with this email address already exists.")
        return email

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get('password')
        confirm_password = cleaned_data.get('confirm_password')

        if password and confirm_password:
            if password != confirm_password:
                self.add_error('confirm_password', "Passwords do not match.")
            if len(password) < 6:
                self.add_error('password', "Password must contain at least 6 characters.")
        return cleaned_data


class UserLoginForm(forms.Form):
    """
    User login authentication form with remember me option.
    """
    username = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={
            'class': 'form-control form-control-lg',
            'placeholder': 'Enter username or email',
            'autocomplete': 'username',
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control form-control-lg',
            'placeholder': 'Enter password',
            'autocomplete': 'current-password',
        })
    )
    remember_me = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input',
            'id': 'rememberMeCheck'
        })
    )

