import json
import logging
import os
import time
import csv
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.db.models import Q
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_POST

from .models import AnalysisSession, Document, Requirement, ConflictReport, ConflictItem
from .forms import DocumentUploadForm, UserRegisterForm, UserLoginForm
from .services.document_parser import DocumentParser
from .services.embeddings import get_embedding_provider, MockEmbeddingProvider
from .services.similarity import SimilaritySearchService
from .services.conflict_analyzer import get_llm_provider

logger = logging.getLogger(__name__)


def upload_view(request):
    """
    Main landing page and upload view.
    Accepts PDF, DOCX, and TXT files, sets similarity thresholds, and initiates analysis.
    """
    if request.method == 'POST':
        form = DocumentUploadForm(request.POST, request.FILES)
        if form.is_valid():
            title = form.cleaned_data.get('title') or 'Requirements Conflict Analysis'
            threshold = form.cleaned_data.get('similarity_threshold', 0.50)
            if threshold is None:
                threshold = 0.50
            cross_doc = form.cleaned_data.get('cross_document_only', False)

            # Create the analysis session linked to user if logged in
            user = request.user if request.user.is_authenticated else None
            session = AnalysisSession.objects.create(
                user=user,
                title=title,
                similarity_threshold=threshold,
                cross_document_only=cross_doc,
                status='PENDING'
            )

            # Save uploaded document records
            uploaded_files = request.FILES.getlist('documents')
            for f in uploaded_files:
                ext = f.name.lower().rsplit('.', 1)[-1] if '.' in f.name else 'txt'
                Document.objects.create(
                    session=session,
                    file=f,
                    filename=f.name,
                    file_type=ext,
                    file_size=f.size
                )

            # Run extraction and conflict detection pipeline
            try:
                _run_analysis_pipeline(session)
                messages.success(request, f"Analysis completed successfully for '{session.title}'!")
                return redirect('detector:report', session_id=session.id)
            except Exception as e:
                logger.exception("Pipeline execution failure:")
                session.status = 'FAILED'
                session.error_message = str(e)
                session.save(update_fields=['status', 'error_message'])
                messages.error(request, f"Analysis failed: {str(e)}")
                return redirect('detector:report', session_id=session.id)
        else:
            messages.error(request, "Please check the form inputs and ensure valid document files are selected.")
    else:
        form = DocumentUploadForm()

    if request.user.is_authenticated:
        recent_sessions = AnalysisSession.objects.filter(user=request.user).order_by('-created_at')[:8]
    else:
        recent_sessions = AnalysisSession.objects.all().order_by('-created_at')[:8]
    return render(request, 'detector/upload.html', {
        'form': form,
        'recent_sessions': recent_sessions,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Authentication Views
# ─────────────────────────────────────────────────────────────────────────────

def register_view(request):
    """User registration page."""
    if request.user.is_authenticated:
        return redirect('detector:upload')

    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = User.objects.create_user(
                username=form.cleaned_data['username'],
                email=form.cleaned_data['email'],
                password=form.cleaned_data['password'],
            )
            login(request, user)
            messages.success(request, f"Welcome, {user.username}! Your account has been created successfully.")
            return redirect('detector:upload')
        else:
            messages.error(request, "Please correct the errors below to complete registration.")
    else:
        form = UserRegisterForm()

    return render(request, 'detector/register.html', {'form': form})


def login_view(request):
    """User login page."""
    if request.user.is_authenticated:
        return redirect('detector:upload')

    next_url = request.GET.get('next', '')

    if request.method == 'POST':
        form = UserLoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            remember_me = form.cleaned_data.get('remember_me', True)

            # Support login by email too
            if '@' in username:
                try:
                    u = User.objects.get(email__iexact=username)
                    username = u.username
                except User.DoesNotExist:
                    pass

            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                if not remember_me:
                    request.session.set_expiry(0)  # session expires when browser closes
                messages.success(request, f"Welcome back, {user.username}!")
                return redirect(next_url or 'detector:upload')
            else:
                messages.error(request, "Invalid username or password. Please try again.")
    else:
        form = UserLoginForm()

    return render(request, 'detector/login.html', {'form': form, 'next': next_url})


def logout_view(request):
    """Logs the user out and redirects to login page."""
    username = request.user.username if request.user.is_authenticated else ''
    logout(request)
    if username:
        messages.info(request, f"You have been signed out successfully. See you soon, {username}!")
    return redirect('detector:login')


def profile_view(request):
    """User profile page showing their analysis session stats."""
    if not request.user.is_authenticated:
        return redirect('detector:login')

    user_sessions = AnalysisSession.objects.filter(user=request.user).order_by('-created_at')
    total_contradictions = sum(s.contradictions_count for s in user_sessions)
    total_reqs = sum(s.total_requirements for s in user_sessions)

    context = {
        'user_sessions': user_sessions,
        'total_sessions': user_sessions.count(),
        'total_contradictions': total_contradictions,
        'total_requirements': total_reqs,
    }
    return render(request, 'detector/profile.html', context)


def _run_analysis_pipeline(session: AnalysisSession):
    """
    Executes the end-to-end extraction, vector embedding, similarity retrieval,
    and LLM conflict reasoning pipeline for an AnalysisSession.
    """
    start_time = time.time()
    session.status = 'PARSING'
    session.save(update_fields=['status'])

    parser = DocumentParser()
    emb_provider = get_embedding_provider()
    llm_provider = get_llm_provider()

    session.embedding_provider_used = emb_provider.name
    session.llm_provider_used = llm_provider.name
    session.save(update_fields=['embedding_provider_used', 'llm_provider_used'])

    documents = session.documents.all()
    if not documents.exists():
        session.status = 'FAILED'
        session.error_message = "No documents found in session."
        session.save(update_fields=['status', 'error_message'])
        return

    # Phase 1: Extract individual requirements with provenance
    all_requirements = []
    for doc in documents:
        try:
            if doc.file and os.path.exists(doc.file.path):
                parsed_items = parser.parse_file(doc.file.path, doc.filename)
            else:
                doc.file.seek(0)
                parsed_items = parser.parse_file(doc.file, doc.filename)
        except Exception as e:
            logger.error(f"Error parsing {doc.filename}: {e}")
            # Try passing file object directly
            try:
                doc.file.seek(0)
                parsed_items = parser.parse_file(doc.file, doc.filename)
            except Exception as e2:
                logger.error(f"Secondary parsing attempt failed for {doc.filename}: {e2}")
                parsed_items = []

        doc.extracted_count = len(parsed_items)
        doc.save(update_fields=['extracted_count'])

        for idx, item in enumerate(parsed_items, start=1):
            all_requirements.append(Requirement(
                session=session,
                document=doc,
                req_id=item.get('req_id') or f"REQ-{doc.id}-{idx:03d}",
                text=item.get('text', '').strip(),
                page_number=item.get('page_number'),
                section=item.get('section') or '',
                order_index=idx
            ))

    if not all_requirements:
        session.status = 'FAILED'
        session.error_message = "No requirements could be extracted from the provided files. Please ensure documents contain requirement text."
        session.save(update_fields=['status', 'error_message'])
        return

    # Bulk insert requirements
    Requirement.objects.bulk_create(all_requirements)
    saved_reqs = list(session.requirements.all().select_related('document'))
    session.total_requirements = len(saved_reqs)
    session.status = 'EMBEDDING'
    session.save(update_fields=['total_requirements', 'status'])

    # Phase 2: Compute vector embeddings
    req_texts = [r.text for r in saved_reqs]
    try:
        embeddings = emb_provider.get_embeddings(req_texts)
        for req, emb in zip(saved_reqs, embeddings):
            req.set_embedding(emb)
        Requirement.objects.bulk_update(saved_reqs, ['embedding_json'])
    except Exception as e:
        logger.warning(f"Primary embedding failed ({e}), falling back to deterministic mock embeddings.")
        mock_emb = MockEmbeddingProvider()
        embeddings = mock_emb.get_embeddings(req_texts)
        for req, emb in zip(saved_reqs, embeddings):
            req.set_embedding(emb)
        Requirement.objects.bulk_update(saved_reqs, ['embedding_json'])

    # Phase 3: Semantic similarity candidate pair search
    session.status = 'ANALYZING'
    session.save(update_fields=['status'])

    sim_service = SimilaritySearchService()
    candidate_pairs = sim_service.find_candidate_pairs(
        saved_reqs,
        threshold=session.similarity_threshold,
        cross_document_only=session.cross_document_only,
        max_candidates=150
    )

    session.total_pairs_compared = len(candidate_pairs)
    session.save(update_fields=['total_pairs_compared'])

    # Create ConflictReport instance
    report, _ = ConflictReport.objects.get_or_create(session=session)
    report.conflicts.all().delete()  # Clear existing items if re-analyzing

    # Phase 4: LLM reasoning and classification of candidate pairs
    contradictions_count = 0
    overlaps_count = 0
    related_count = 0
    items_to_create = []

    for req_a, req_b, sim_score in candidate_pairs:
        meta_a = {
            'doc_name': req_a.document.filename,
            'page': req_a.page_number,
            'section': req_a.section,
            'req_id': req_a.req_id,
        }
        meta_b = {
            'doc_name': req_b.document.filename,
            'page': req_b.page_number,
            'section': req_b.section,
            'req_id': req_b.req_id,
        }

        analysis = llm_provider.analyze_pair(req_a.text, req_b.text, meta_a, meta_b)
        raw_classification = str(analysis.get('classification', 'RELATED')).upper()
        if raw_classification not in ('CONTRADICTORY', 'OVERLAPPING', 'RELATED'):
            raw_classification = 'RELATED'

        if raw_classification == 'CONTRADICTORY':
            contradictions_count += 1
        elif raw_classification == 'OVERLAPPING':
            overlaps_count += 1
        else:
            related_count += 1

        items_to_create.append(ConflictItem(
            report=report,
            req_a=req_a,
            req_b=req_b,
            classification=raw_classification,
            conflict_type=analysis.get('conflict_type') or 'General Conflict',
            confidence=float(analysis.get('confidence', 0.85)),
            similarity_score=sim_score,
            explanation=analysis.get('explanation') or 'Potential semantic interaction between requirement specifications.',
            suggested_resolution=analysis.get('suggested_resolution') or 'Review both statements with system stakeholders to unify definitions.'
        ))

    if items_to_create:
        ConflictItem.objects.bulk_create(items_to_create)

    # Wrap up session & report
    total_conflicts = contradictions_count + overlaps_count
    if contradictions_count > 0:
        summary = f"{contradictions_count} Direct Contradiction{'s' if contradictions_count > 1 else ''} Identified"
    elif overlaps_count > 0:
        summary = f"{overlaps_count} Requirement Overlap{'s' if overlaps_count > 1 else ''} Identified"
    else:
        summary = "No Contradictions Found"

    exec_time = round(time.time() - start_time, 2)
    report.status_summary = summary
    report.has_conflicts = total_conflicts > 0
    report.execution_time_seconds = exec_time
    report.save()

    session.contradictions_count = contradictions_count
    session.overlaps_count = overlaps_count
    session.related_count = related_count
    session.status = 'COMPLETED'
    session.save()


def report_view(request, session_id: int):
    """
    Renders the detailed Conflict Report page.
    Includes KPI stat cards, filter tabs, live search, and full side-by-side comparison cards.
    """
    session = get_object_or_404(AnalysisSession, id=session_id)
    report = getattr(session, 'report', None)

    filter_type = request.GET.get('filter', 'all').lower()
    query = request.GET.get('q', '').strip()

    conflicts = report.conflicts.all().select_related('req_a__document', 'req_b__document') if report else ConflictItem.objects.none()

    if filter_type in ('contradictory', 'contradictions'):
        conflicts = conflicts.filter(classification='CONTRADICTORY')
    elif filter_type in ('overlapping', 'overlaps'):
        conflicts = conflicts.filter(classification='OVERLAPPING')
    elif filter_type in ('related', 'compatible'):
        conflicts = conflicts.filter(classification='RELATED')

    if query:
        conflicts = conflicts.filter(
            Q(req_a__text__icontains=query) |
            Q(req_b__text__icontains=query) |
            Q(conflict_type__icontains=query) |
            Q(explanation__icontains=query) |
            Q(suggested_resolution__icontains=query) |
            Q(req_a__req_id__icontains=query) |
            Q(req_b__req_id__icontains=query)
        )

    all_conflicts_count = report.conflicts.count() if report else 0
    contradictions_count = session.contradictions_count
    overlaps_count = session.overlaps_count
    related_count = session.related_count

    context = {
        'session': session,
        'report': report,
        'conflicts': conflicts,
        'filter_type': filter_type,
        'query': query,
        'all_conflicts_count': all_conflicts_count,
        'contradictions_count': contradictions_count,
        'overlaps_count': overlaps_count,
        'related_count': related_count,
        'documents': session.documents.all(),
        'requirements': session.requirements.all().select_related('document')[:100],
    }
    return render(request, 'detector/report.html', context)


def session_list_view(request):
    """
    Displays a history of all analyzed sessions.
    """
    sessions = AnalysisSession.objects.all().order_by('-created_at')
    return render(request, 'detector/session_list.html', {'sessions': sessions})


def load_sample_view(request):
    """
    One-click sample project loader.
    Populates Hospital SRS v1 and Outpatient Billing BRD v2, executes detection,
    and lands the user immediately on the interactive conflict report.
    """
    sample_dir = settings.BASE_DIR / 'sample_data'
    srs_path = sample_dir / 'sample_srs_hospital_v1.txt'
    brd_path = sample_dir / 'sample_brd_billing_v2.txt'

    if not srs_path.exists() or not brd_path.exists():
        messages.error(request, "Sample files not found on disk.")
        return redirect('detector:upload')

    with open(srs_path, 'r', encoding='utf-8') as f:
        srs_content = f.read()
    with open(brd_path, 'r', encoding='utf-8') as f:
        brd_content = f.read()

    # Create session with threshold tuned for TF-IDF vector similarity
    session = AnalysisSession.objects.create(
        title="Sample: Healthcare SRS v1.0 vs Outpatient Billing BRD v2.0",
        similarity_threshold=0.10,
        cross_document_only=True,
        status='PENDING'
    )

    doc_srs = Document.objects.create(
        session=session,
        filename='sample_srs_hospital_v1.txt',
        file_type='txt',
        file_size=len(srs_content.encode('utf-8'))
    )
    doc_srs.file.save('sample_srs_hospital_v1.txt', ContentFile(srs_content.encode('utf-8')))

    doc_brd = Document.objects.create(
        session=session,
        filename='sample_brd_billing_v2.txt',
        file_type='txt',
        file_size=len(brd_content.encode('utf-8'))
    )
    doc_brd.file.save('sample_brd_billing_v2.txt', ContentFile(brd_content.encode('utf-8')))

    try:
        _run_analysis_pipeline(session)
        messages.success(request, "Sample project loaded and analyzed successfully! Contradictions detected between SRS and BRD.")
    except Exception as e:
        logger.exception("Failed to run sample analysis:")
        messages.error(request, f"Sample analysis error: {e}")

    return redirect('detector:report', session_id=session.id)


@require_POST
def delete_session_view(request, session_id: int):
    """
    Deletes an analysis session, removing its records and associated files.
    """
    session = get_object_or_404(AnalysisSession, id=session_id)
    title = session.title
    # Delete uploaded files on disk
    for doc in session.documents.all():
        if doc.file:
            try:
                if os.path.exists(doc.file.path):
                    os.remove(doc.file.path)
            except Exception:
                pass
    session.delete()
    messages.info(request, f"Session '{title}' has been deleted.")
    return redirect('detector:session_list')


def export_report_view(request, session_id: int, format: str):
    """
    Exports the conflict report as JSON or CSV.
    """
    session = get_object_or_404(AnalysisSession, id=session_id)
    report = getattr(session, 'report', None)

    if not report:
        return HttpResponse("No report generated for this session", status=404)

    conflicts = report.conflicts.all().select_related('req_a__document', 'req_b__document')

    if format == 'json':
        data = {
            'session_id': session.id,
            'title': session.title,
            'created_at': session.created_at.isoformat(),
            'status_summary': report.status_summary,
            'llm_provider': session.llm_provider_used,
            'embedding_provider': session.embedding_provider_used,
            'total_requirements': session.total_requirements,
            'contradictions_count': session.contradictions_count,
            'overlaps_count': session.overlaps_count,
            'related_count': session.related_count,
            'conflicts': [
                {
                    'classification': c.classification,
                    'conflict_type': c.conflict_type,
                    'confidence': c.confidence,
                    'similarity_score': c.similarity_score,
                    'explanation': c.explanation,
                    'suggested_resolution': c.suggested_resolution,
                    'req_a': {
                        'id': c.req_a.req_id,
                        'document': c.req_a.document.filename,
                        'page': c.req_a.page_number,
                        'section': c.req_a.section,
                        'text': c.req_a.text,
                    },
                    'req_b': {
                        'id': c.req_b.req_id,
                        'document': c.req_b.document.filename,
                        'page': c.req_b.page_number,
                        'section': c.req_b.section,
                        'text': c.req_b.text,
                    }
                }
                for c in conflicts
            ]
        }
        response = HttpResponse(json.dumps(data, indent=2), content_type='application/json')
        response['Content-Disposition'] = f'attachment; filename="conflict_report_session_{session.id}.json"'
        return response

    elif format == 'csv':
        response = HttpResponse(content_type='text/csv; charset=utf-8')
        response['Content-Disposition'] = f'attachment; filename="conflict_report_session_{session.id}.csv"'
        writer = csv.writer(response)
        writer.writerow([
            'Classification', 'Conflict Type', 'Confidence', 'Similarity',
            'Req A ID', 'Req A Document', 'Req A Page', 'Req A Section', 'Req A Text',
            'Req B ID', 'Req B Document', 'Req B Page', 'Req B Section', 'Req B Text',
            'Explanation', 'Suggested Resolution'
        ])
        for c in conflicts:
            writer.writerow([
                c.classification,
                c.conflict_type,
                f"{c.confidence:.2f}",
                f"{c.similarity_score:.2f}",
                c.req_a.req_id,
                c.req_a.document.filename,
                c.req_a.page_number or '',
                c.req_a.section or '',
                c.req_a.text,
                c.req_b.req_id,
                c.req_b.document.filename,
                c.req_b.page_number or '',
                c.req_b.section or '',
                c.req_b.text,
                c.explanation,
                c.suggested_resolution
            ])
        return response

    return HttpResponse("Unsupported export format. Use 'json' or 'csv'.", status=400)
