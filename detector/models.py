import json
from django.conf import settings
from django.db import models
from django.urls import reverse

class AnalysisSession(models.Model):
    """
    Represents an analysis session or project where one or more
    requirements documents are analyzed for contradictions and overlaps.
    """
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('PARSING', 'Extracting Requirements'),
        ('EMBEDDING', 'Generating Semantic Embeddings'),
        ('ANALYZING', 'Analyzing Pairs with LLM'),
        ('COMPLETED', 'Analysis Completed'),
        ('FAILED', 'Analysis Failed'),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='analysis_sessions'
    )
    title = models.CharField(max_length=255, default='Requirements Analysis Session')
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    error_message = models.TextField(blank=True, default='')
    similarity_threshold = models.FloatField(default=0.50)
    cross_document_only = models.BooleanField(
        default=False,
        help_text="If True, only compare requirements across different documents."
    )
    llm_provider_used = models.CharField(max_length=100, blank=True, default='')
    embedding_provider_used = models.CharField(max_length=100, blank=True, default='')
    
    total_requirements = models.IntegerField(default=0)
    total_pairs_compared = models.IntegerField(default=0)
    contradictions_count = models.IntegerField(default=0)
    overlaps_count = models.IntegerField(default=0)
    related_count = models.IntegerField(default=0)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.title} (#{self.id}) - {self.status}"

    def get_absolute_url(self):
        return reverse('detector:report', kwargs={'session_id': self.id})

    @property
    def has_conflicts(self):
        return self.contradictions_count > 0 or self.overlaps_count > 0


class Document(models.Model):
    """
    An uploaded requirement document (PDF, DOCX, or TXT).
    """
    session = models.ForeignKey(AnalysisSession, on_delete=models.CASCADE, related_name='documents')
    file = models.FileField(upload_to='requirements_docs/%Y/%m/')
    filename = models.CharField(max_length=255)
    file_type = models.CharField(max_length=20) # pdf, docx, txt
    file_size = models.BigIntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    extracted_count = models.IntegerField(default=0)

    class Meta:
        ordering = ['uploaded_at']

    def __str__(self):
        return f"{self.filename} ({self.file_type.upper()})"

    @property
    def file_size_display(self):
        if self.file_size < 1024:
            return f"{self.file_size} B"
        elif self.file_size < 1024 * 1024:
            return f"{self.file_size / 1024:.1f} KB"
        else:
            return f"{self.file_size / (1024 * 1024):.2f} MB"


class Requirement(models.Model):
    """
    An individual requirement extracted from an uploaded document,
    maintaining page number and section provenance.
    """
    session = models.ForeignKey(AnalysisSession, on_delete=models.CASCADE, related_name='requirements')
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='requirements')
    req_id = models.CharField(max_length=100, default='REQ')
    text = models.TextField()
    page_number = models.IntegerField(null=True, blank=True)
    section = models.CharField(max_length=255, blank=True, null=True, default='')
    embedding_json = models.TextField(blank=True, null=True)
    order_index = models.IntegerField(default=0)

    class Meta:
        ordering = ['document', 'order_index', 'id']

    def __str__(self):
        return f"[{self.req_id}] {self.document.filename}: {self.text[:60]}..."

    @property
    def reference_display(self):
        parts = [f"Doc: {self.document.filename}"]
        if self.page_number is not None:
            parts.append(f"Page {self.page_number}")
        if self.section and self.section.strip():
            parts.append(f"Section: {self.section.strip()}")
        return " | ".join(parts)

    def set_embedding(self, vec):
        if vec is not None:
            self.embedding_json = json.dumps(list(vec))
        else:
            self.embedding_json = None

    def get_embedding(self):
        if self.embedding_json:
            try:
                return json.loads(self.embedding_json)
            except Exception:
                return None
        return None


class ConflictReport(models.Model):
    """
    Report summarizing all conflicts detected for an analysis session.
    """
    session = models.OneToOneField(AnalysisSession, on_delete=models.CASCADE, related_name='report')
    created_at = models.DateTimeField(auto_now_add=True)
    status_summary = models.CharField(max_length=255, default='No Contradictions Found')
    has_conflicts = models.BooleanField(default=False)
    execution_time_seconds = models.FloatField(default=0.0)

    def __str__(self):
        return f"Conflict Report for Session #{self.session.id}: {self.status_summary}"


class ConflictItem(models.Model):
    """
    A specific detected conflict or relationship between two requirements.
    """
    CLASSIFICATION_CHOICES = [
        ('CONTRADICTORY', 'Contradictory'),
        ('OVERLAPPING', 'Overlapping or Duplicate'),
        ('RELATED', 'Related but Not Conflicting'),
    ]

    report = models.ForeignKey(ConflictReport, on_delete=models.CASCADE, related_name='conflicts')
    req_a = models.ForeignKey(Requirement, on_delete=models.CASCADE, related_name='conflicts_as_a')
    req_b = models.ForeignKey(Requirement, on_delete=models.CASCADE, related_name='conflicts_as_b')
    
    classification = models.CharField(max_length=30, choices=CLASSIFICATION_CHOICES)
    conflict_type = models.CharField(max_length=150, default='General Conflict')
    confidence = models.FloatField(default=0.0) # 0.0 - 1.0
    similarity_score = models.FloatField(default=0.0)
    
    explanation = models.TextField(help_text="Clear explanation of the exact conflict or overlap")
    suggested_resolution = models.TextField(help_text="Suggested engineering or business resolution")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = [
            models.Case(
                models.When(classification='CONTRADICTORY', then=0),
                models.When(classification='OVERLAPPING', then=1),
                models.When(classification='RELATED', then=2),
                default=3
            ),
            '-confidence',
            '-similarity_score'
        ]

    def __str__(self):
        return f"{self.classification}: {self.req_a.req_id} vs {self.req_b.req_id} ({self.conflict_type})"

    @property
    def confidence_percent(self):
        return int(round(self.confidence * 100))

    @property
    def similarity_percent(self):
        return int(round(self.similarity_score * 100))

    @property
    def badge_class(self):
        if self.classification == 'CONTRADICTORY':
            return 'danger'
        elif self.classification == 'OVERLAPPING':
            return 'warning'
        return 'success'
