import os
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile

from .models import AnalysisSession, Document, Requirement, ConflictReport, ConflictItem
from .services.document_parser import DocumentParser
from .services.embeddings import LocalTfidfEmbeddingProvider, MockEmbeddingProvider, get_embedding_provider
from .services.similarity import cosine_similarity, SimilaritySearchService
from .services.conflict_analyzer import SafeMockLLMProvider, get_llm_provider


class DocumentParserTest(TestCase):
    def setUp(self):
        self.parser = DocumentParser()

    def test_parse_txt_requirements(self):
        content = (
            "REQ-AUTH-01: The system shall terminate sessions after 15 minutes of inactivity.\n\n"
            "REQ-AUTH-02: All users must authenticate via multi-factor authentication (MFA).\n\n"
            "REQ-PERF-01: Patient search queries shall return within 1.0 second."
        )
        items = self.parser.parse_file(content, 'test_requirements.txt')
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0]['req_id'], 'REQ-AUTH-01')
        self.assertIn('15 minutes', items[0]['text'])

    def test_parse_docx_file(self):
        docx_path = os.path.join('sample_data', 'sample_srs_hospital_v1.docx')
        if os.path.exists(docx_path):
            items = self.parser.parse_file(docx_path, 'sample_srs_hospital_v1.docx')
            self.assertGreater(len(items), 0)
            self.assertTrue(any('REQ-AUTH' in item['req_id'] or '15 minutes' in item['text'] for item in items))


class EmbeddingAndSimilarityTest(TestCase):
    def test_local_tfidf_embeddings(self):
        provider = LocalTfidfEmbeddingProvider()
        texts = [
            "The system shall terminate user sessions after 15 minutes of inactivity.",
            "User login sessions shall automatically expire after 30 minutes of idle time.",
            "The database shall store patient clinical notes encrypted using AES-256."
        ]
        vectors = provider.get_embeddings(texts)
        self.assertEqual(len(vectors), 3)
        
        # Test similarity: texts 0 and 1 are both about session timeouts, should have higher similarity than text 2
        sim_0_1 = cosine_similarity(vectors[0], vectors[1])
        sim_0_2 = cosine_similarity(vectors[0], vectors[2])
        self.assertGreater(sim_0_1, sim_0_2)

    def test_similarity_search_candidate_pairs(self):
        session = AnalysisSession.objects.create(title="Test Search")
        doc = Document.objects.create(session=session, filename="test.txt", file_type="txt")
        
        req1 = Requirement.objects.create(
            session=session, document=doc, req_id="REQ-1",
            text="User session timeout is 15 minutes."
        )
        req2 = Requirement.objects.create(
            session=session, document=doc, req_id="REQ-2",
            text="Clinician idle session timeout is 30 minutes."
        )
        
        provider = MockEmbeddingProvider()
        vecs = provider.get_embeddings([req1.text, req2.text])
        req1.set_embedding(vecs[0])
        req2.set_embedding(vecs[1])
        req1.save()
        req2.save()
        
        service = SimilaritySearchService()
        pairs = service.find_candidate_pairs([req1, req2], threshold=0.10)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0], req1)
        self.assertEqual(pairs[0][1], req2)


class SafeMockLLMConflictTest(TestCase):
    def setUp(self):
        self.analyzer = SafeMockLLMProvider()

    def test_detect_numeric_conflict(self):
        text_a = "The system shall automatically terminate user sessions after 15 minutes of inactivity."
        text_b = "Clinician sessions shall automatically expire after 30 minutes of idle inactivity."
        result = self.analyzer.analyze_pair(text_a, text_b)
        self.assertEqual(result['classification'], 'CONTRADICTORY')
        self.assertIn('Numerical', result['conflict_type'])

    def test_detect_security_mfa_conflict(self):
        text_a = "All healthcare practitioners shall authenticate using mandatory multi-factor authentication (MFA)."
        text_b = "Medical administrative staff may access patient billing histories using single-factor password authentication without requiring multi-factor authentication."
        result = self.analyzer.analyze_pair(text_a, text_b)
        self.assertEqual(result['classification'], 'CONTRADICTORY')
        self.assertIn('Authentication Security', result['conflict_type'])

    def test_detect_hosting_architecture_conflict(self):
        text_a = "All database clusters and services shall be hosted on-premise within the hospital air-gapped data center."
        text_b = "All enterprise services shall be migrated to 100% public cloud architecture."
        result = self.analyzer.analyze_pair(text_a, text_b)
        self.assertEqual(result['classification'], 'CONTRADICTORY')


class WebFlowIntegrationTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_upload_page_loads(self):
        response = self.client.get(reverse('detector:upload'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Requirements Contradiction Detector')
        self.assertContains(response, 'Drag and drop files here')

    def test_load_sample_flow(self):
        response = self.client.get(reverse('detector:load_sample'), follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'CONTRADICTION')
        self.assertContains(response, 'Healthcare SRS')

        # Check database records created
        session = AnalysisSession.objects.first()
        self.assertIsNotNone(session)
        self.assertEqual(session.status, 'COMPLETED')
        self.assertGreater(session.total_requirements, 5)
        self.assertGreater(session.contradictions_count, 0)

        # Test export JSON
        json_resp = self.client.get(reverse('detector:export_report', args=[session.id, 'json']))
        self.assertEqual(json_resp.status_code, 200)
        self.assertEqual(json_resp['Content-Type'], 'application/json')

        # Test export CSV
        csv_resp = self.client.get(reverse('detector:export_report', args=[session.id, 'csv']))
        self.assertEqual(csv_resp.status_code, 200)
        self.assertIn('text/csv', csv_resp['Content-Type'])

    def test_history_list_page(self):
        session = AnalysisSession.objects.create(title="History Test")
        response = self.client.get(reverse('detector:session_list'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "History Test")
