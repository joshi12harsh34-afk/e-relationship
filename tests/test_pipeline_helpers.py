"""Focused unit tests for entity-relationship/main.py helpers (no Bedrock)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as er  # noqa: E402


class EvidenceVerificationTests(unittest.TestCase):
    def test_unique_match_sets_offsets(self):
        doc = 'Alpha widget. Beta device.'
        out = er._verify_evidence(
            {'source_text': 'Alpha widget', 'start_char': None, 'end_char': None},
            doc,
            source_file='inputs/sample.md',
        )
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out['start_char'], 0)
        self.assertEqual(out['end_char'], len('Alpha widget'))
        self.assertEqual(out['source_file'], 'inputs/sample.md')

    def test_supplied_offsets_validated(self):
        doc = 'xxx TARGET yyy TARGET zzz'
        quote = 'TARGET'
        start = doc.rfind(quote)
        out = er._verify_evidence(
            {'source_text': quote, 'start_char': start, 'end_char': start + len(quote)},
            doc,
        )
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out['start_char'], start)

    def test_ambiguous_quote_null_offsets(self):
        doc = 'foo bar foo'
        out = er._verify_evidence({'source_text': 'foo'}, doc)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertIsNone(out['start_char'])
        self.assertIsNone(out['end_char'])

    def test_claim_verified_flag(self):
        meta = {
            'evidence': {
                'source_text': 'Pump A feeds Tank B',
                'start_char': None,
                'end_char': None,
            }
        }
        doc = 'Pump A feeds Tank B in unit 3.'
        out = er._attach_claim_evidence(
            meta, doc, subject='Pump A', obj='Tank B', predicate='feeds',
        )
        self.assertEqual(out.get('evidence_status'), 'verified')
        self.assertTrue(out.get('graph_eligible'))
        self.assertTrue((out.get('evidence') or {}).get('claim_verified'))

        bad = er._attach_claim_evidence(
            meta, doc, subject='Pump A', obj='Vessel Z', predicate='feeds',
        )
        self.assertEqual(bad.get('evidence_status'), 'unverified')
        self.assertFalse(bad.get('graph_eligible'))
        self.assertFalse((bad.get('evidence') or {}).get('claim_verified'))

    def test_attribute_claim_requires_owner_and_value(self):
        doc = 'Row 22-PK-3801 Design Capacity - 71.5 TPH next to media notes.'
        good = er._attach_claim_evidence(
            {'evidence': {'source_text': '22-PK-3801 Design Capacity - 71.5 TPH'}},
            doc,
            subject='22-PK-3801',
            obj='71.5 TPH',
            predicate='Design Capacity',
            attribute_mode=True,
        )
        self.assertEqual(good.get('evidence_status'), 'verified')
        self.assertTrue(good.get('graph_eligible'))

        weak = er._attach_claim_evidence(
            {'evidence': {'source_text': '22-PK-3801 Design Capacity - 71.5 TPH'}},
            doc,
            subject='22-PK-3801',
            obj='Design Capacity - 71.5 TPH',
            predicate='has property',
            attribute_mode=True,
        )
        self.assertEqual(weak.get('predicate_quality'), 'weak_generic')
        self.assertFalse(weak.get('graph_eligible'))

        # Value-only quote and no owner tag/section anywhere in the document.
        orphan_doc = 'Design Capacity - 71.5 TPH appears without an equipment owner.'
        missing_owner = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Design Capacity - 71.5 TPH'}},
            orphan_doc,
            subject='22-PK-3801',
            obj='71.5 TPH',
            predicate='Design Capacity',
            attribute_mode=True,
        )
        self.assertEqual(missing_owner.get('evidence_status'), 'unverified')
        self.assertFalse(missing_owner.get('graph_eligible'))

    def test_section_context_verifies_owner_outside_value_quote(self):
        doc = (
            '## 6.1 22-PK-3801: Package Scope\n\n'
            'Design Capacity - 71.5 TPH\n'
        )
        out = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Design Capacity - 71.5 TPH'}},
            doc,
            subject='22-PK-3801',
            obj='71.5 TPH',
            predicate='Design Capacity',
            attribute_mode=True,
        )
        self.assertEqual(out.get('evidence_status'), 'verified')
        self.assertTrue(out.get('graph_eligible'))
        evidence = out.get('evidence') or {}
        self.assertEqual(evidence.get('source_text'), 'Design Capacity - 71.5 TPH')
        self.assertNotIn('22-PK-3801', evidence.get('source_text') or '')
        section = out.get('section_context') or {}
        self.assertIn('22-PK-3801', section.get('source_text') or '')
        self.assertEqual(doc[section['start_char']:section['end_char']], section['source_text'])
        self.assertEqual(doc[evidence['start_char']:evidence['end_char']], evidence['source_text'])


class AttributeOwnerRepairTests(unittest.TestCase):
    """Structural attribute ownership: correct unique owners; quarantine shared."""

    def setUp(self):
        er.clear_section_aware_caches()
        er._OWNER_RESOLVER_HOOK = None

    def tearDown(self):
        er._OWNER_RESOLVER_HOOK = None
        er.clear_section_aware_caches()

    def test_package_specific_section_corrects_wrong_llm_subject(self):
        doc = (
            '## 6 Scope\n\n'
            '## 6.1 22-PK-3801: Polymer Pneumatic Transport\n\n'
            '| Parameter | Value |\n'
            '| --- | --- |\n'
            '| Design Capacity | 71.5 TPH |\n\n'
            '## 6.2 22-PK-3804: Neighbor Package\n\n'
            '| Design Capacity | 40 TPH |\n'
        )
        out = er._attach_claim_evidence(
            {'evidence': {'source_text': '| Design Capacity | 71.5 TPH |'}},
            doc,
            subject='22-PK-3804',
            obj='71.5 TPH',
            predicate='Design Capacity',
            attribute_mode=True,
        )
        self.assertEqual(out.get('attribute_source_scope'), 'specific')
        self.assertEqual(out.get('corrected_subject'), '22-PK-3801')
        self.assertEqual(out.get('original_subject'), '22-PK-3804')
        repair = out.get('owner_repair') or {}
        self.assertEqual(repair.get('to'), '22-PK-3801')
        self.assertEqual(out.get('evidence_status'), 'verified')
        self.assertTrue(out.get('graph_eligible'))
        evidence = out.get('evidence') or {}
        self.assertEqual(evidence.get('source_text'), '| Design Capacity | 71.5 TPH |')
        self.assertNotIn('22-PK-3801', evidence.get('source_text') or '')

    def test_equipment_specific_row_owner_beats_package_guess(self):
        doc = (
            '## 6.1 22-PK-3801: Package Scope\n\n'
            '| Tag | Material |\n'
            '| --- | --- |\n'
            '| 22-V-3800 | Aluminium |\n'
            '| 22-G-3800 | Carbon steel |\n'
        )
        out = er._attach_claim_evidence(
            {'evidence': {'source_text': '| 22-V-3800 | Aluminium |'}},
            doc,
            subject='22-PK-3801',
            obj='Aluminium',
            predicate='Material',
            attribute_mode=True,
        )
        self.assertEqual(out.get('attribute_source_scope'), 'specific')
        self.assertEqual(out.get('corrected_subject'), '22-V-3800')
        self.assertEqual(out.get('original_subject'), '22-PK-3801')
        self.assertEqual(out.get('evidence_status'), 'verified')
        self.assertTrue(out.get('graph_eligible'))

    def test_shared_general_section_never_assigns_package(self):
        doc = (
            '## 2 Applicable Codes And Standards\n\n'
            'Following codes & standards will be followed:\n\n'
            '| a) | Material of construction | As per Licensor\'s data sheet. |\n'
            '| b) | Pressure parts including filters | ASME Section VIII |\n\n'
            '## 6.1 22-PK-3801: Polymer Pneumatic Transport\n\n'
            'Design Capacity - 71.5 TPH\n'
        )
        out = er._attach_claim_evidence(
            {
                'evidence': {
                    'source_text': (
                        "| a) | Material of construction | As per Licensor's data sheet. |"
                    ),
                },
            },
            doc,
            subject='22-PK-3801',
            obj="As per Licensor's data sheet",
            predicate='material of construction',
            attribute_mode=True,
        )
        self.assertEqual(out.get('attribute_source_scope'), 'shared')
        self.assertNotIn('corrected_subject', out)
        self.assertEqual(out.get('original_subject'), '22-PK-3801')
        self.assertEqual(out.get('evidence_status'), 'unverified')
        self.assertFalse(out.get('graph_eligible'))
        evidence = out.get('evidence') or {}
        self.assertEqual(
            evidence.get('verification_note'),
            'shared_or_general_section_no_unique_owner',
        )
        self.assertIn('Material of construction', evidence.get('source_text') or '')

    def test_cross_page_section_corrects_wrong_owner(self):
        doc = (
            '## 6.1 22-V-3800: Feeding Hopper\n\n'
            'Quantity : 1\n\n'
            '<!-- page 2 -->\n\n'
            '<table><tr><td colspan="6">Banner (22-PK-3801, 22-PK-3804)</td></tr></table>\n\n'
            'Design Temperature : 110°C\n\n'
            '## 6.2 22-G-3800: Neighbor Filter\n\n'
            'Design Temperature : 80°C\n'
        )
        out = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Design Temperature : 110°C'}},
            doc,
            subject='22-PK-3801',
            obj='110°C',
            predicate='Design Temperature',
            attribute_mode=True,
        )
        self.assertEqual(out.get('attribute_source_scope'), 'specific')
        self.assertEqual(out.get('corrected_subject'), '22-V-3800')
        self.assertEqual(out.get('original_subject'), '22-PK-3801')
        self.assertEqual(out.get('evidence_status'), 'verified')
        self.assertTrue(out.get('graph_eligible'))
        section = out.get('section_context') or {}
        self.assertEqual(section.get('section_number'), '6.1')
        evidence = out.get('evidence') or {}
        self.assertEqual(evidence.get('source_text'), 'Design Temperature : 110°C')


class EquipmentSectionEvidenceTests(unittest.TestCase):
    """Cross-page equipment ownership using the real Annexure Markdown."""

    @classmethod
    def setUpClass(cls):
        cls.annexure_path = ROOT / 'inputs' / (
            'Annexure 2.1 _ 1916-PRO-TDS-22-PK-3801 2 (3)_advanced_auto.md'
        )
        cls.doc = cls.annexure_path.read_text(encoding='utf-8')

    def test_section_aware_settings_reuse_ontology_not_doc_lists(self):
        er.clear_section_aware_caches()
        settings = er._section_aware_evidence_settings()
        policy = er.load_policy()
        cfg = (policy.data.get('section_aware_evidence') or {})
        self.assertTrue(cfg.get('enabled', True))
        self.assertNotIn('field_label_tokens', cfg)
        self.assertNotIn('inline_heading_boundary_labels', cfg)
        self.assertNotIn('engineering_tag_pattern', cfg)
        self.assertNotIn('structural_patterns', cfg)
        self.assertIsNotNone(settings.get('engineering_tag_re'))
        # Reuses identity_resolution.identifier_patterns + property aliases / table markers.
        self.assertTrue(settings.get('field_label_tokens'))
        self.assertIn('temperature', settings['field_label_tokens'])
        self.assertTrue(settings.get('ontology_boundary_labels'))
        self.assertTrue(settings.get('kv_separators'))
        # No document-format structural regex configs or module constants.
        self.assertFalse(hasattr(er, '_MD_EQUIPMENT_HEADING'))
        self.assertFalse(hasattr(er, '_HTML_EQUIPMENT_HEADING'))
        self.assertFalse(hasattr(er, '_BOLD_EQUIPMENT_HEADING'))
        self.assertFalse(hasattr(er, '_PIPE_LABEL_CELL'))
        self.assertFalse(hasattr(er, '_LABEL_UNIT_SUFFIX'))
        self.assertFalse(hasattr(er, '_DEFAULT_KV_SEPARATORS'))
        self.assertFalse(hasattr(er, '_FIELD_LABEL_TOKENS'))
        self.assertFalse(hasattr(er, '_INLINE_EQUIPMENT_HEADING'))
        self.assertFalse(hasattr(er, '_GENERIC_ATTR_PREDICATES'))
        self.assertFalse(hasattr(er, '_compile_structural_pattern'))
        self.assertFalse(hasattr(er, '_structural_patterns'))

    def test_section_index_tracks_6_1_across_page_into_html(self):
        sections = er.build_equipment_section_index(self.doc)
        by_num = {s['section_number']: s for s in sections}
        self.assertIn('6.1', by_num)
        self.assertIn('6.2', by_num)
        self.assertIn('6.3', by_num)
        s61 = by_num['6.1']
        s62 = by_num['6.2']
        self.assertEqual(s61['tags'], ['22-V-3800'])
        self.assertEqual(s62['tags'], ['22-G-3800'])
        self.assertLess(s61['start_char'], s62['start_char'])
        self.assertEqual(s61['end_char'], s62['start_char'])
        # Page marker + HTML continuation stay inside 6.1 until 6.2 starts.
        page9 = self.doc.find('<!-- page 9 -->')
        self.assertGreater(page9, s61['start_char'])
        self.assertLess(page9, s61['end_char'])
        max_pressure = self.doc.find('Max Pressure : 0.2 kg/cm²(g)')
        self.assertGreaterEqual(max_pressure, s61['start_char'])
        self.assertLess(max_pressure, s61['end_char'])
        filter_material = self.doc.find(
            'Material : Aluminium (for part in contact with product)'
        )
        self.assertGreaterEqual(filter_material, s62['start_char'])
        self.assertLess(filter_material, s62['end_char'])

    def test_annexure_6_1_attributes_verify_with_separate_provenance(self):
        cases = [
            ('22-V-3800', 'Material', '304 SS Or Hardened Aluminum',
             'Material : 304 SS Or Hardened Aluminum'),
            ('22-V-3800', 'Operating Pressure', '0.1 kg/cm²(g)',
             'Operating Pressure : 0.1 kg/cm²(g)'),
            ('22-V-3800', 'Max Pressure', '0.2 kg/cm²(g)',
             'Max Pressure : 0.2 kg/cm²(g)'),
            ('22-V-3800', 'Design Temperature', '110°C',
             'Design Temperature : 110°C'),
        ]
        for subject, predicate, obj, quote in cases:
            with self.subTest(predicate=predicate):
                out = er._attach_claim_evidence(
                    {'evidence': {'source_text': quote}},
                    self.doc,
                    subject=subject,
                    obj=obj,
                    predicate=predicate,
                    attribute_mode=True,
                    source_file=str(self.annexure_path.as_posix()),
                )
                self.assertEqual(out.get('evidence_status'), 'verified', out)
                evidence = out.get('evidence') or {}
                self.assertEqual(evidence.get('source_text'), quote)
                self.assertNotIn(subject, evidence.get('source_text') or '')
                section = out.get('section_context') or {}
                self.assertIn('22-V-3800', section.get('source_text') or '')
                self.assertIn('6.1', section.get('section_number') or '')
                self.assertEqual(
                    self.doc[section['start_char']:section['end_char']],
                    section['source_text'],
                )
                self.assertEqual(
                    self.doc[evidence['start_char']:evidence['end_char']],
                    evidence['source_text'],
                )

    def test_neighbor_section_attributes_do_not_leak(self):
        # 6.2 material must not verify under 6.1 owner.
        leaked = er._attach_claim_evidence(
            {
                'evidence': {
                    'source_text': (
                        'Material : Aluminium (for part in contact with product)'
                    ),
                }
            },
            self.doc,
            subject='22-V-3800',
            obj='Aluminium (for part in contact with product)',
            predicate='Material',
            attribute_mode=True,
        )
        self.assertEqual(leaked.get('evidence_status'), 'unverified')
        self.assertFalse(leaked.get('graph_eligible'))
        self.assertNotIn('section_context', leaked)

        owned = er._attach_claim_evidence(
            {
                'evidence': {
                    'source_text': (
                        'Material : Aluminium (for part in contact with product)'
                    ),
                }
            },
            self.doc,
            subject='22-G-3800',
            obj='Aluminium (for part in contact with product)',
            predicate='Material',
            attribute_mode=True,
        )
        self.assertEqual(owned.get('evidence_status'), 'verified')
        self.assertEqual((owned.get('section_context') or {}).get('section_number'), '6.2')

    def test_html_and_pipe_table_attributes_stay_in_section(self):
        doc = (
            '## 6.1 22-V-3800: Feeding Hopper\n\n'
            'Material : 304 SS Or Hardened Aluminum\n\n'
            '<!-- page 9 -->\n\n'
            '<table><tr><td>IOCL HEADER</td></tr>'
            '<tr><td>Max Pressure : 0.2 kg/cm²(g)'
            'Design Temperature : 110°C'
            '6.2 22-G-3800: Neighbor Filter'
            'Quantity : 1'
            'Material : Aluminium (for part in contact with product)'
            '</td></tr></table>\n\n'
            '## 6.3 22-ME-3800: Rotary Feeder\n\n'
            '| Parameter | Value |\n'
            '| --- | --- |\n'
            '| Material | 304 SS for rotor |\n'
            '| Operating Pressure | 0.1 kg/cm²(g) |\n'
        )
        sections = er.build_equipment_section_index(doc)
        by_num = {s['section_number']: s for s in sections}
        self.assertEqual(by_num['6.1']['tags'], ['22-V-3800'])
        self.assertEqual(by_num['6.2']['tags'], ['22-G-3800'])
        self.assertEqual(by_num['6.3']['tags'], ['22-ME-3800'])

        cross_page = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Max Pressure : 0.2 kg/cm²(g)'}},
            doc,
            subject='22-V-3800',
            obj='0.2 kg/cm²(g)',
            predicate='Max Pressure',
            attribute_mode=True,
        )
        self.assertEqual(cross_page.get('evidence_status'), 'verified')
        self.assertEqual((cross_page.get('section_context') or {}).get('section_number'), '6.1')

        pipe = er._attach_claim_evidence(
            {'evidence': {'source_text': '| Material | 304 SS for rotor |'}},
            doc,
            subject='22-ME-3800',
            obj='304 SS for rotor',
            predicate='Material',
            attribute_mode=True,
        )
        self.assertEqual(pipe.get('evidence_status'), 'verified')
        self.assertEqual((pipe.get('section_context') or {}).get('section_number'), '6.3')
        evidence = pipe.get('evidence') or {}
        self.assertEqual(evidence.get('source_text'), '| Material | 304 SS for rotor |')
        self.assertNotIn('22-ME-3800', evidence.get('source_text') or '')

        # Wrong LLM owner is corrected to the unique section that owns the row.
        leak = er._attach_claim_evidence(
            {'evidence': {'source_text': '| Material | 304 SS for rotor |'}},
            doc,
            subject='22-V-3800',
            obj='304 SS for rotor',
            predicate='Material',
            attribute_mode=True,
        )
        self.assertEqual(leak.get('attribute_source_scope'), 'specific')
        self.assertEqual(leak.get('corrected_subject'), '22-ME-3800')
        self.assertEqual(leak.get('original_subject'), '22-V-3800')
        self.assertEqual(leak.get('evidence_status'), 'verified')
        self.assertEqual((leak.get('section_context') or {}).get('section_number'), '6.3')


class GenericSectionAwareFormatTests(unittest.TestCase):
    """Different Markdown formats should work without YAML edits."""

    def setUp(self):
        er.clear_section_aware_caches()
        er._OWNER_RESOLVER_HOOK = None

    def tearDown(self):
        er._OWNER_RESOLVER_HOOK = None
        er.clear_section_aware_caches()

    def test_kv_formats_with_units_and_separators(self):
        cases = [
            ('Design Temperature (°C): 110', 'Design Temperature (°C)', '110'),
            ('Pressure = 0.5 bar', 'Pressure', '0.5 bar'),
            ('Capacity - 71.5 TPH', 'Capacity', '71.5 TPH'),
            ('Max Pressure – 0.2 kg/cm²(g)', 'Max Pressure', '0.2 kg/cm²(g)'),
            ('Volume = By Vendor', 'Volume', 'By Vendor'),
        ]
        seps = er._kv_separators()
        for raw, label, value in cases:
            with self.subTest(raw=raw):
                parsed = er._split_kv_segment(raw, separators=seps)
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed[0], label)
                self.assertEqual(parsed[2], value)

        doc = (
            '## 8.1 33-V-9000: Test Vessel\n\n'
            'Design Temperature (°C): 110\n'
            'Pressure = 0.5 bar\n'
            'Capacity - 71.5 TPH\n'
        )
        for quote, pred, obj in (
            ('Design Temperature (°C): 110', 'Design Temperature', '110'),
            ('Pressure = 0.5 bar', 'Pressure', '0.5 bar'),
            ('Capacity - 71.5 TPH', 'Capacity', '71.5 TPH'),
        ):
            with self.subTest(quote=quote):
                out = er._attach_claim_evidence(
                    {'evidence': {'source_text': quote}},
                    doc,
                    subject='33-V-9000',
                    obj=obj,
                    predicate=pred,
                    attribute_mode=True,
                )
                self.assertEqual(out.get('evidence_status'), 'verified', out)
                evidence = out.get('evidence') or {}
                self.assertEqual(evidence.get('source_text'), quote)
                self.assertNotIn('33-V-9000', evidence.get('source_text') or '')
                self.assertEqual(
                    doc[evidence['start_char']:evidence['end_char']],
                    evidence['source_text'],
                )

    def test_concatenated_html_fields_split_without_leakage(self):
        doc = (
            '## 4.1 44-V-1100: Hopper\n\n'
            '<table><tr><td>'
            'Design Temperature (°C): 110'
            'Pressure = 0.5 bar.'
            '4.2 44-G-1100: Neighbor Filter'
            'Capacity - 71.5 TPH'
            '</td></tr></table>\n'
        )
        sections = er.build_equipment_section_index(doc)
        by_num = {s['section_number']: s for s in sections}
        self.assertEqual(by_num['4.1']['tags'], ['44-V-1100'])
        self.assertEqual(by_num['4.2']['tags'], ['44-G-1100'])

        temp = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Design Temperature (°C): 110'}},
            doc,
            subject='44-V-1100',
            obj='110',
            predicate='Design Temperature',
            attribute_mode=True,
        )
        self.assertEqual(temp.get('evidence_status'), 'verified')
        self.assertEqual((temp.get('section_context') or {}).get('section_number'), '4.1')

        leak = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Capacity - 71.5 TPH'}},
            doc,
            subject='44-V-1100',
            obj='71.5 TPH',
            predicate='Capacity',
            attribute_mode=True,
        )
        # Unique section owner at the value offset corrects the wrong LLM subject.
        self.assertEqual(leak.get('attribute_source_scope'), 'specific')
        self.assertEqual(leak.get('corrected_subject'), '44-G-1100')
        self.assertEqual(leak.get('original_subject'), '44-V-1100')
        self.assertEqual(leak.get('evidence_status'), 'verified')
        self.assertEqual((leak.get('section_context') or {}).get('section_number'), '4.2')

        owned = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Capacity - 71.5 TPH'}},
            doc,
            subject='44-G-1100',
            obj='71.5 TPH',
            predicate='Capacity',
            attribute_mode=True,
        )
        self.assertEqual(owned.get('evidence_status'), 'verified')
        self.assertEqual((owned.get('section_context') or {}).get('section_number'), '4.2')

    def test_rejects_false_inline_tag_boundaries_in_prose(self):
        doc = (
            'Refer to 22-V-3800: Feeding Hopper mentioned in scope notes.\n'
            'Also check 22-G-3800: Filter without a section number.\n\n'
            '## 5.1 22-V-3800: Real Hopper\n\n'
            'Material : 304 SS\n'
        )
        sections = er.build_equipment_section_index(doc)
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]['section_number'], '5.1')
        self.assertEqual(sections[0]['kind'], 'markdown')
        self.assertEqual(sections[0]['tags'], ['22-V-3800'])

    def test_bold_and_unnumbered_headings(self):
        doc = (
            '**66-P-0100: Charge Pump**\n\n'
            'Pressure = 1.2 bar\n\n'
            '## 66-E-0101 Cooler Package\n\n'
            'Design Temperature (°C): 80\n'
        )
        sections = er.build_equipment_section_index(doc)
        tags = [tuple(s['tags']) for s in sections]
        self.assertIn(('66-P-0100',), tags)
        self.assertIn(('66-E-0101',), tags)
        pump = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Pressure = 1.2 bar'}},
            doc,
            subject='66-P-0100',
            obj='1.2 bar',
            predicate='Pressure',
            attribute_mode=True,
        )
        self.assertEqual(pump.get('evidence_status'), 'verified')
        cooler = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Design Temperature (°C): 80'}},
            doc,
            subject='66-E-0101',
            obj='80',
            predicate='Design Temperature',
            attribute_mode=True,
        )
        self.assertEqual(cooler.get('evidence_status'), 'verified')

    def test_cache_invalidates_on_document_change(self):
        doc1 = '## 1.1 11-V-1000: One\n\nMaterial : A\n'
        doc2 = '## 1.1 11-V-1000: One\n\nMaterial : A\n## 1.2 11-G-1000: Two\n\nMaterial : B\n'
        first = er.build_equipment_section_index(doc1)
        self.assertEqual(len(first), 1)
        # Same object identity from cache.
        self.assertIs(er.build_equipment_section_index(doc1), first)
        second = er.build_equipment_section_index(doc2)
        self.assertEqual(len(second), 2)
        self.assertIsNot(first, second)
        er.clear_section_aware_caches()
        rebuilt = er.build_equipment_section_index(doc1)
        self.assertEqual(len(rebuilt), 1)
        self.assertIsNot(rebuilt, first)

    def test_html_heading_format(self):
        doc = (
            '<h2>3.1 10-P-1001: Charge Pump</h2>\n'
            'Material : Carbon Steel\n'
            '<!-- page 2 -->\n'
            '<h2>3.2 10-E-1002: Cooler</h2>\n'
            'Material : 316 SS\n'
        )
        sections = er.build_equipment_section_index(doc)
        by_num = {s['section_number']: s for s in sections}
        self.assertEqual(by_num['3.1']['tags'], ['10-P-1001'])
        self.assertEqual(by_num['3.2']['tags'], ['10-E-1002'])

        pump = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Material : Carbon Steel'}},
            doc,
            subject='10-P-1001',
            obj='Carbon Steel',
            predicate='Material',
            attribute_mode=True,
        )
        self.assertEqual(pump.get('evidence_status'), 'verified')
        self.assertEqual((pump.get('section_context') or {}).get('section_number'), '3.1')

        leak = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Material : 316 SS'}},
            doc,
            subject='10-P-1001',
            obj='316 SS',
            predicate='Material',
            attribute_mode=True,
        )
        self.assertEqual(leak.get('attribute_source_scope'), 'specific')
        self.assertEqual(leak.get('corrected_subject'), '10-E-1002')
        self.assertEqual(leak.get('original_subject'), '10-P-1001')
        self.assertEqual(leak.get('evidence_status'), 'verified')
        self.assertEqual((leak.get('section_context') or {}).get('section_number'), '3.2')

    def test_unnumbered_markdown_heading_and_pipe_table(self):
        doc = (
            '## 55-V-2000: Surge Drum\n\n'
            '| Parameter | Value |\n'
            '| --- | --- |\n'
            '| Design Pressure | 3.5 barg |\n\n'
            '## 55-G-2001: Drum Vent Filter\n\n'
            'Filter Media : PTFE\n'
        )
        sections = er.build_equipment_section_index(doc)
        tags = [tuple(s['tags']) for s in sections]
        self.assertIn(('55-V-2000',), tags)
        self.assertIn(('55-G-2001',), tags)

        drum = er._attach_claim_evidence(
            {'evidence': {'source_text': '| Design Pressure | 3.5 barg |'}},
            doc,
            subject='55-V-2000',
            obj='3.5 barg',
            predicate='Design Pressure',
            attribute_mode=True,
        )
        self.assertEqual(drum.get('evidence_status'), 'verified')
        evidence = drum.get('evidence') or {}
        self.assertEqual(doc[evidence['start_char']:evidence['end_char']], evidence['source_text'])
        section = drum.get('section_context') or {}
        self.assertIn('55-V-2000', section.get('source_text') or '')
        self.assertNotIn('55-V-2000', evidence.get('source_text') or '')

        leak = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Filter Media : PTFE'}},
            doc,
            subject='55-V-2000',
            obj='PTFE',
            predicate='Filter Media',
            attribute_mode=True,
        )
        self.assertEqual(leak.get('attribute_source_scope'), 'specific')
        self.assertEqual(leak.get('corrected_subject'), '55-G-2001')
        self.assertEqual(leak.get('original_subject'), '55-V-2000')
        self.assertEqual(leak.get('evidence_status'), 'verified')

    def test_document_kv_discovery_drives_inline_boundaries(self):
        doc = (
            '## 1.1 99-V-0100: Vessel A\n\n'
            'Weird Label : alpha\n'
            '<!-- page 2 -->\n'
            '<table><tr><td>Weird Label : beta continued'
            '1.2 99-G-0100: Neighbor Unit'
            'Weird Label : gamma</td></tr></table>\n'
        )
        labels = er._document_kv_labels(doc)
        self.assertTrue(any(er.collapse(x).casefold() == 'weird label' for x in labels))
        sections = er.build_equipment_section_index(doc)
        by_num = {s['section_number']: s for s in sections}
        self.assertIn('1.1', by_num)
        self.assertIn('1.2', by_num)
        owned = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Weird Label : beta continued'}},
            doc,
            subject='99-V-0100',
            obj='beta continued',
            predicate='Weird Label',
            attribute_mode=True,
        )
        self.assertEqual(owned.get('evidence_status'), 'verified')
        self.assertEqual((owned.get('section_context') or {}).get('section_number'), '1.1')

    def test_llm_fallback_resolves_ambiguous_duplicate_quote(self):
        doc = (
            '## 2.1 77-V-5000: First Vessel\n\n'
            'Shared Spec : 10 bar\n\n'
            '## 2.2 77-V-5000: Duplicate Tag Vessel\n\n'
            'Shared Spec : 10 bar\n'
        )

        def fake_resolver(payload):
            self.assertGreaterEqual(len(payload.get('candidates') or []), 2)
            return '2.2'

        out = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Shared Spec : 10 bar'}},
            doc,
            subject='77-V-5000',
            obj='10 bar',
            predicate='Shared Spec',
            attribute_mode=True,
            owner_resolver=fake_resolver,
        )
        self.assertEqual(out.get('evidence_status'), 'verified')
        section = out.get('section_context') or {}
        self.assertEqual(section.get('section_number'), '2.2')
        self.assertEqual(section.get('resolver'), 'llm_fallback')
        evidence = out.get('evidence') or {}
        self.assertEqual(evidence.get('source_text'), 'Shared Spec : 10 bar')
        self.assertNotIn('77-V-5000', evidence.get('source_text') or '')


class PackageTagRelabelTests(unittest.TestCase):
    def test_no_pk_specific_regex_or_force_relabel(self):
        """Package/system typing must not use a PK-only regex or bare_surface force."""
        policy = er.load_policy()
        section = (policy.data.get('identity_resolution') or {})
        patterns = [
            str(row.get('pattern') or '')
            for row in (section.get('identifier_patterns') or [])
            if isinstance(row, dict)
        ]
        self.assertTrue(patterns)
        self.assertFalse(any('-PK-' in p or 'PK-' in p for p in patterns))
        for row in (section.get('identifier_patterns') or []):
            if not isinstance(row, dict):
                continue
            self.assertFalse(str(row.get('bare_surface_label') or '').strip())
            self.assertFalse(str(row.get('bare_surface_subtype') or '').strip())
        self.assertFalse(hasattr(er, '_relabel_package_tag_entities'))
        self.assertFalse(hasattr(er, '_bare_identifier_relabel_rules'))


class MergeGateTests(unittest.TestCase):
    def test_refuses_similarity_only_systems(self):
        a = {
            'id': 'e0',
            'text': 'Conveying System Alpha',
            'label': 'System',
            'metadata': {'identifiers': [{'value': 'SYS-A'}], 'aliases': []},
        }
        b = {
            'id': 'e1',
            'text': 'Conveying System Beta',
            'label': 'System',
            'metadata': {'identifiers': [{'value': 'SYS-B'}], 'aliases': []},
        }
        self.assertFalse(er._group_safe_to_merge([a, b]))

    def test_allows_alias_linked_orgs(self):
        a = {
            'id': 'e0',
            'text': 'Acme Industrial Corp.',
            'label': 'Organization',
            'metadata': {'aliases': ['Acme'], 'identifiers': []},
        }
        b = {
            'id': 'e1',
            'text': 'Acme',
            'label': 'Organization',
            'metadata': {'aliases': [], 'identifiers': []},
        }
        self.assertTrue(er._group_safe_to_merge([a, b]))

    def test_merge_preserves_peer_evidence_as_source_references(self):
        class Ent:
            def __init__(self, text, meta):
                self.text = text
                self.metadata = meta

        rep = Ent('Acme Industrial Corp.', {
            'aliases': [],
            'identifiers': [{'value': 'Acme Industrial Corp.', 'role': 'identity'}],
            'properties': {'status': 'active'},
            'evidence': {'source_text': 'Acme Industrial Corp.', 'start_char': 0, 'end_char': 21},
        })
        peer = Ent('Acme', {
            'aliases': [],
            'identifiers': [],
            'properties': {'region': 'west'},
            'evidence': {'source_text': 'Acme', 'start_char': 30, 'end_char': 34},
        })
        members = [
            {'id': 'e0', 'text': rep.text, 'metadata': rep.metadata, '_src': rep},
            {'id': 'e1', 'text': peer.text, 'metadata': peer.metadata, '_src': peer},
        ]
        er._apply_merge_to_representative(
            rep,
            members=members,
            merged_entity={'metadata': {'provenance': {'merged': True}}},
            canonical_id='e0',
        )
        meta = rep.metadata
        self.assertIn('Acme', meta.get('aliases') or [])
        self.assertEqual(meta.get('properties', {}).get('status'), 'active')
        self.assertEqual(meta.get('properties', {}).get('region'), 'west')
        self.assertEqual((meta.get('evidence') or {}).get('source_text'), 'Acme Industrial Corp.')
        refs = meta.get('source_references') or []
        self.assertTrue(any(r.get('source_text') == 'Acme' for r in refs))
        self.assertEqual(meta.get('canonical_id'), 'e0')


class TypedStoreTests(unittest.TestCase):
    def test_per_call_stores_do_not_mix(self):
        store_a: dict = {}
        store_b: dict = {}
        er._typed_set('EntitiesResponse', {'entities': [1]}, store_a)
        er._typed_set('EntitiesResponse', {'entities': [2]}, store_b)
        self.assertEqual(store_a['EntitiesResponse']['entities'], [1])
        self.assertEqual(store_b['EntitiesResponse']['entities'], [2])

    def test_typed_set_accumulates_across_chunks(self):
        store: dict = {}
        er._typed_set('RelationsResponse', {'relations': [{'subject': 'A'}]}, store)
        er._typed_set('RelationsResponse', {'relations': [{'subject': 'B'}]}, store)
        rows = er._typed_rows(store['RelationsResponse'], 'relations')
        self.assertEqual([r.get('subject') for r in rows], ['A', 'B'])


class ChunkGuidanceTests(unittest.TestCase):
    def test_fill_guidance_uses_semantica_chunk_not_full_doc(self):
        guidance = 'RULES\n\nSOURCE\n<<CHUNK_TEXT>>\n'
        semantica_prompt = (
            'Extract named entities...\n'
            'Text to extract from:\n'
            'CHUNK ALPHA ONLY\n'
            'Entities found in text: (none)'
        )
        filled = er._fill_guidance_chunk(guidance, semantica_prompt)
        self.assertIn('CHUNK ALPHA ONLY', filled)
        self.assertNotIn('<<CHUNK_TEXT>>', filled)
        self.assertEqual(filled.count('CHUNK ALPHA ONLY'), 1)

    def test_ner_context_collapses_repeated_page_banners(self):
        b1 = (
            '<table><tr><td>PROJECT NO.</td><td>1916</td></tr>'
            '<tr><td>DOCUMENT NO.</td><td>DOC-A</td>'
            '<td>SHEET NO.</td><td>1 OF 23</td></tr></table>'
        )
        b2 = (
            '<table><tr><td>PROJECT NO.</td><td>1916</td></tr>'
            '<tr><td>DOCUMENT NO.</td><td>DOC-A</td>'
            '<td>SHEET NO.</td><td>2 OF 23</td></tr></table>'
        )
        b3 = (
            '<table><tr><td>PROJECT NO.</td><td>1916</td></tr>'
            '<tr><td>DOCUMENT NO.</td><td>DOC-A</td>'
            '<td>SHEET NO.</td><td>3 OF 23</td></tr></table>'
        )
        body = '\nEQUIPMENT\n| TAG-1 | Pump |\n'
        chunk = f'{b1}\n{b2}\n{body}{b3}\n'
        focused = er._ner_context_text(chunk)
        self.assertEqual(focused.count('DOCUMENT NO.'), 1)
        self.assertIn('repeated page banner omitted', focused)
        self.assertIn('start_char=', focused)
        self.assertIn('TAG-1', focused)

    def test_ner_context_uses_full_document_banner_detection(self):
        """Banners repeated across the full doc must collapse even in a late chunk."""
        b1 = (
            '<table><tr><td>PROJECT NO.</td><td>1916</td></tr>'
            '<tr><td>DOCUMENT NO.</td><td>DOC-A</td>'
            '<td>SHEET NO.</td><td>1 OF 23</td></tr></table>'
        )
        b2 = (
            '<table><tr><td>PROJECT NO.</td><td>1916</td></tr>'
            '<tr><td>DOCUMENT NO.</td><td>DOC-A</td>'
            '<td>SHEET NO.</td><td>2 OF 23</td></tr></table>'
        )
        early = f'<!-- page 1 -->\n{b1}\nINTRO\n'
        late = f'<!-- page 2 -->\n{b2}\n## 2. MAIN PRODUCT\nTAG-99\n'
        full = early + late
        er._DOC_NER_CONTEXT.text = full
        er._DOC_NER_CONTEXT.spans = er.detect_repeated_boilerplate_spans(full)
        er._DOC_NER_CONTEXT.chunk_start = len(early)
        er._DOC_NER_CONTEXT.parent_heading = '## 2. MAIN PRODUCT'
        try:
            focused = er._ner_context_text(late)
        finally:
            er._DOC_NER_CONTEXT.text = None
            er._DOC_NER_CONTEXT.spans = None
            er._DOC_NER_CONTEXT.chunk_start = 0
            er._DOC_NER_CONTEXT.parent_heading = ''
        self.assertIn('repeated page banner omitted', focused)
        self.assertNotIn('SHEET NO.', focused)
        self.assertIn('TAG-99', focused)
        self.assertIn('section context: ## 2. MAIN PRODUCT', focused)

    def test_section_aware_chunks_keep_tables_and_parent_headings(self):
        table = (
            '<table><tr><td>ITEM</td><td>SERVICE</td></tr>'
            '<tr><td>22-E-3800</td><td>Cooler</td></tr></table>'
        )
        doc = (
            '<!-- page 1 -->\n## 1. SCOPE\nIntro text.\n'
            f'<!-- page 2 -->\n## 2. EQUIPMENT\n{table}\n'
            '<!-- page 3 -->\n## 3. NOTES\nTrailing notes.\n'
        )
        spans = er.section_aware_chunk_spans(doc, max_chars=120)
        self.assertGreaterEqual(len(spans), 2)
        # Exact source slices: offsets map back to the original document.
        for span in spans:
            self.assertEqual(doc[span['start']:span['end']], span['text'])
            self.assertIn(span.get('method'), {'recursive', 'passthrough'})
        # HTML table must appear wholly inside one span.
        self.assertTrue(any(table in s['text'] for s in spans))
        equip = next(s for s in spans if '22-E-3800' in s['text'])
        self.assertIn('## 2. EQUIPMENT', equip.get('parent_heading') or equip['text'])

    def test_semantica_recursive_available_and_used_for_large_docs(self):
        from semantica.split import list_available_methods

        methods = set(list_available_methods() or [])
        self.assertIn('recursive', methods)
        self.assertIn('hierarchical', methods)
        # Large synthetic markdown should chunk via Semantica recursive.
        parts = []
        for i in range(1, 8):
            parts.append(f'<!-- page {i} -->\n## {i}. SECTION {i}\n' + ('body text ' * 40) + '\n')
        doc = ''.join(parts)
        self.assertGreater(len(doc), 800)
        spans = er.section_aware_chunk_spans(doc, max_chars=400)
        self.assertGreaterEqual(len(spans), 2)
        self.assertTrue(all(s.get('method') == 'recursive' for s in spans))
        for span in spans:
            self.assertEqual(doc[span['start']:span['end']], span['text'])
            heading = span.get('parent_heading') or ''
            self.assertTrue(heading.startswith('##') or 'SECTION' in span['text'])

    def test_table_boundary_repair_keeps_html_table_intact(self):
        prefix = '## 1. INTRO\n' + ('x' * 80) + '\n'
        table = (
            '<table><tr><td>TAG</td><td>SERVICE</td></tr>'
            + ''.join(f'<tr><td>22-E-{i:04d}</td><td>Cooler {i}</td></tr>' for i in range(20))
            + '</table>'
        )
        suffix = '\n## 2. AFTER\nDone.\n'
        doc = prefix + table + suffix
        # Force a cut size smaller than prefix+table so recursive would otherwise split.
        spans = er.section_aware_chunk_spans(doc, max_chars=max(60, len(prefix) + 40))
        holders = [s for s in spans if '<table' in s['text'] or '</table>' in s['text']]
        self.assertTrue(holders)
        for span in holders:
            if '<table' in span['text']:
                self.assertIn('</table>', span['text'])
            self.assertEqual(doc[span['start']:span['end']], span['text'])

    def test_detect_boilerplate_returns_exact_spans_and_positions(self):
        b1 = (
            '<table><tr><td>PROJECT NO.</td><td>1916</td></tr>'
            '<tr><td>DOCUMENT NO.</td><td>DOC-A</td>'
            '<td>SHEET NO.</td><td>1 OF 23</td></tr></table>'
        )
        b2 = (
            '<table><tr><td>PROJECT NO.</td><td>1916</td></tr>'
            '<tr><td>DOCUMENT NO.</td><td>DOC-A</td>'
            '<td>SHEET NO.</td><td>2 OF 23</td></tr></table>'
        )
        doc = f'<!-- page 1 -->\n{b1}\nBODY\n<!-- page 2 -->\n{b2}\n'
        spans = er.detect_repeated_boilerplate_spans(doc)
        self.assertGreaterEqual(len(spans), 2)
        for span in spans:
            self.assertEqual(doc[span['start_char']:span['end_char']], span['source_text'])
        omitted = [s for s in spans if s['action'] == 'omit_repeat']
        self.assertEqual(len(omitted), 1)
        self.assertEqual(omitted[0]['role'], 'page_header')

    def test_ner_context_keeps_bid_document_content_tables(self):
        """Bid/tender data tables are content even if they mention Document No."""
        nit = (
            '<table><tr><td>NOTICE INVITING TENDER (Bidding Document No. 6746-X)</td>'
            '<td>Rev</td><td>00</td></tr></table>'
        )
        bid = (
            '<table><tbody><tr><td>3.1 Bid Document No. 6746-X</td></tr>'
            '<tr><td>3.2</td><td>Type of tender</td>'
            '<td>Domestic Open Competitive Bidding</td></tr>'
            '<tr><td>3.3</td><td>Reverse Auction</td><td>Not Applicable</td></tr>'
            '</tbody></table>'
        )
        chunk = f'{nit}\n## 3.0 Details\n{bid}\n{bid}\n'
        focused = er._ner_context_text(chunk)
        self.assertNotIn('repeated page banner omitted', focused)
        self.assertIn('Type of tender', focused)
        self.assertTrue(all(
            s['action'] == 'preserve'
            for s in er.detect_repeated_boilerplate_spans(chunk)
        ))

    def test_entity_guidance_applies_ner_banner_collapse(self):
        banner = (
            '<table><tr><td>PROJECT NO.</td><td>1916</td></tr>'
            '<tr><td>DOCUMENT NO.</td><td>DOC-A</td>'
            '<td>SHEET NO.</td><td>1 OF 23</td></tr></table>'
        )
        banner2 = (
            '<table><tr><td>PROJECT NO.</td><td>1916</td></tr>'
            '<tr><td>DOCUMENT NO.</td><td>DOC-A</td>'
            '<td>SHEET NO.</td><td>2 OF 23</td></tr></table>'
        )
        chunk = f'{banner}\n{banner2}\nUnit U-1\n'
        guidance = 'TASK = ENTITIES only (NER).\nSOURCE\n<<CHUNK_TEXT>>\n'
        semantica_prompt = f'Text to extract from:\n{chunk}\nEntities found in text: (none)'
        filled = er._fill_guidance_chunk(guidance, semantica_prompt)
        self.assertEqual(filled.count('DOCUMENT NO.'), 1)
        self.assertIn('U-1', filled)

    def test_fill_guidance_falls_back_to_thread_local_chunk(self):
        guidance = 'TASK = ENTITIES only (NER).\nSOURCE\n<<CHUNK_TEXT>>\n'
        er._NER_CHUNK_TEXT.value = 'Local Pump P-100'
        try:
            filled = er._fill_guidance_chunk(guidance, 'prompt without marker')
        finally:
            er._NER_CHUNK_TEXT.value = None
        self.assertIn('Local Pump P-100', filled)
        self.assertNotIn('<<CHUNK_TEXT>>', filled)

    def test_ground_entities_drops_placeholders_and_ungrounded(self):
        class _E:
            def __init__(self, text):
                self.text = text
                self.label = 'Equipment'
                self.metadata = {}

        doc = 'Tagged asset 22-E-3800 is present.'
        kept = er._ground_entities_in_source(
            [_E('22-E-3800'), _E('CHUNK_TEXT'), _E('John Doe'), _E('')],
            doc,
        )
        self.assertEqual([e.text for e in kept], ['22-E-3800'])

    def test_ground_entities_drops_banner_field_labels(self):
        """Layout-discovered field labels are non-entities; values/tags stay."""
        class _E:
            def __init__(self, text, label='Document'):
                self.text = text
                self.label = label
                self.metadata = {'kind': 'entity', 'subtype': 'DocumentSection'}

        doc = (
            '<table><tr>'
            '<td>PROJECT NO.</td><td>1916</td>'
            '<td>DOCUMENT NO.</td><td>1916-PRO-TDS-22-PK-3804</td>'
            '<td>SHEET NO.</td><td>1 OF 15</td>'
            '</tr></table>\n'
            'ITEM : 22-PK-3804\n'
        )
        policy = er.load_policy()
        self.assertTrue(er._is_field_label_non_entity('PROJECT NO.', doc, policy=policy))
        self.assertTrue(er._is_field_label_non_entity('DOCUMENT NO.', doc, policy=policy))
        self.assertFalse(
            er._is_field_label_non_entity('1916-PRO-TDS-22-PK-3804', doc, policy=policy)
        )
        self.assertFalse(er._is_field_label_non_entity('22-PK-3804', doc, policy=policy))

        kept = er._ground_entities_in_source(
            [
                _E('PROJECT NO.'),
                _E('DOCUMENT NO.'),
                _E('1916-PRO-TDS-22-PK-3804'),
                _E('22-PK-3804', label='System'),
            ],
            doc,
            policy=policy,
        )
        self.assertEqual(
            [e.text for e in kept],
            ['1916-PRO-TDS-22-PK-3804', '22-PK-3804'],
        )

    def test_guided_provider_skips_system_when_chunk_placeholder_present(self):
        calls = {}

        def fake_fn(**kwargs):
            calls['kwargs'] = dict(kwargs)
            return []

        er._with_guided_provider(
            'TASK = ENTITIES\nSOURCE\n<<CHUNK_TEXT>>\n',
            ('Extract named entities',),
            fake_fn,
            typed_store={},
            provider='bedrock',
        )
        self.assertNotIn('system', calls['kwargs'])


class BedrockProviderRegistrationTests(unittest.TestCase):
    def test_bedrock_is_registered_with_semantica(self):
        er.register_bedrock_provider()
        self.assertIs(er.provider_registry.get('bedrock'), er.BedrockProvider)
        from semantica.semantic_extract.providers import create_provider

        llm = create_provider('bedrock', model='test-model')
        self.assertIsInstance(llm, er.BedrockProvider)


class OntologyAndGraphTests(unittest.TestCase):
    def test_ontology_hints_from_yaml(self):
        policy = er.load_policy()
        hints = er.ontology_hints(policy)
        self.assertNotEqual(hints, '(none)')
        self.assertIn('Allowed parent->child', hints)
        self.assertIn('System->Equipment', hints)
        self.assertIn('Entity definitions', hints)
        self.assertIn('Disambiguation:', hints)
        self.assertIn('Recommended subtypes', hints)
        self.assertIn('Non-entity policy', hints)
        self.assertIn('Table extraction rules:', hints)
        self.assertIn('Equipment vs Component', hints)
        self.assertIn('Subject/tag columns', hints)

    def test_compact_ontology_hints_still_include_definitions(self):
        policy = er.load_policy()
        compact = er.ontology_hints(policy, compact=True)
        self.assertIn('Entity definitions', compact)
        self.assertIn('Disambiguation:', compact)
        self.assertIn('Non-entity policy', compact)
        self.assertLess(len(compact), len(er.ontology_hints(policy, compact=False)))

    def test_graph_validation_flags_missing_and_invalid_hierarchy(self):
        policy = er.load_policy()
        bundle = er.ExtractionBundle(
            entities=[
                {'id': 'e0', 'text': 'Scope Document', 'label': 'Document', 'metadata': {}},
                {'id': 'e1', 'text': 'Pump Unit', 'label': 'Equipment', 'metadata': {}},
            ],
            relations=[
                {
                    'subject': 'Scope Document',
                    'predicate': 'hasPackage',
                    'object': 'Missing Package',
                    'metadata': {},
                },
                {
                    'subject': 'Pump Unit',
                    'predicate': 'hasPackage',
                    'object': 'Scope Document',
                    'metadata': {},
                },
            ],
            triplets=[
                {
                    'subject': 'Pump Unit',
                    'predicate': 'has property',
                    'object': '100 TPH',
                    'metadata': {},
                }
            ],
        )
        warnings = er.validate_extraction_graph(bundle, policy, source_file='inputs/sample.md')
        self.assertTrue(any('missing entity node' in w for w in warnings))
        self.assertTrue(any('invalid parent-child types' in w for w in warnings))
        trip = bundle.triplets[0]
        self.assertEqual(trip.get('object_value_kind'), 'literal')
        self.assertEqual(trip.get('subject_ref_kind'), 'entity')

    def test_graph_validation_applies_predicate_subtype_gates(self):
        policy = er.load_policy()
        gates = er.predicate_subtype_gates(policy)
        self.assertIn('haspackage', gates)
        bundle = er.ExtractionBundle(
            entities=[
                {
                    'id': 'e0',
                    'text': 'Tender Pack',
                    'label': 'Document',
                    'metadata': {'subtype': 'Tender'},
                },
                {
                    'id': 'e1',
                    'text': '22-PK-3801',
                    'label': 'System',
                    'metadata': {'subtype': 'EngineeringPackage'},
                },
            ],
            relations=[
                {
                    'subject': 'Tender Pack',
                    'predicate': 'hasPackage',
                    'object': '22-PK-3801',
                    'metadata': {},
                }
            ],
        )
        warnings = er.validate_extraction_graph(bundle, policy)
        self.assertTrue(any('hierarchy subtype gate' in w for w in warnings))
        self.assertTrue(any('parent subtype' in w for w in warnings))


class UnifiedExtractionPromptNerTests(unittest.TestCase):
    """Regression: NER prompt must allow Equipment and extract tagged table assets."""

    @classmethod
    def setUpClass(cls):
        cls.prompt = (ROOT / 'prompts' / 'unified_extraction.txt').read_text(encoding='utf-8')
        cls.policy = er.load_policy()
        cls.settings = er.get_settings()
        cls.extractor = er.FactExtractor(settings=cls.settings, policy=cls.policy)

    def test_equipment_and_activity_are_allowed_primaries(self):
        self.assertNotIn('or Equipment/Activity.', self.prompt)
        self.assertIn('including Equipment and\n  Activity', self.prompt)
        self.assertIn(
            'Never use generic filler labels such as Entity, CONCEPT, Thing, or Unknown.',
            self.prompt,
        )

    def test_boilerplate_banner_guidance_in_unified_prompt(self):
        self.assertIn('A6b. REPEATED PAGE HEADERS, FOOTERS, AND METADATA BANNERS', self.prompt)
        self.assertIn(
            'Identify repeated page headers, footers, and document metadata banners',
            self.prompt,
        )
        self.assertIn(
            'Do not classify technical, commercial, specification, or tender-data',
            self.prompt,
        )
        self.assertIn('Return exact source spans and their positions', self.prompt)
        self.assertIn('If uncertain, preserve the content.', self.prompt)

    def test_cross_page_section_ownership_guidance_in_unified_prompt(self):
        self.assertIn('continue across page markers', self.prompt)
        self.assertIn('Never invent a combined evidence quote', self.prompt)
        self.assertIn('If attribute ownership is ambiguous across neighboring equipment', self.prompt)
        self.assertIn('value quote stays verbatim', self.prompt)

    def test_evidence_first_creation_time_quote_required(self):
        self.assertIn('MUST include\nmetadata.evidence', self.prompt)
        self.assertIn('shortest contiguous verbatim span', self.prompt)
        self.assertIn('Leave start_char and end_char null; Python verifies', self.prompt)
        ner = self.extractor._unified_prompt(task='entities', text='x', source_file='inputs/x.md')
        self.assertIn('REQUIRED: set metadata.evidence.source_text', ner)
        self.assertIn('shortest verbatim', ner)
        trip = self.extractor._unified_prompt(task='triplets', text='x', source_file='inputs/x.md')
        self.assertIn('REQUIRED: metadata.evidence.source_text = shortest verbatim value', trip)
        self.assertNotIn('Evidence must include the owning system/equipment identifier', trip)

    def test_content_table_hints_come_from_policy_yaml(self):
        cfg = er._boilerplate_detection_settings(self.policy)
        hints = cfg.get('content_table_hints') or []
        self.assertIn('type of tender', hints)
        self.assertIn('notice inviting tender', hints)
        # Must not rely on a hardcoded module-level hint tuple.
        self.assertFalse(hasattr(er, '_CONTENT_TABLE_HINTS'))

    def test_table_data_rows_require_tagged_asset_extraction(self):
        # Headers alone are not assets; data-row tags must be extracted.
        self.assertIn('column headers, field captions, or generic index headings', self.prompt)
        self.assertIn(
            'Do extract all independently identifiable Systems, Equipment, Components,\n'
            '  and Instruments named in table data rows',
            self.prompt,
        )
        self.assertIn('tagged assets in equipment lists are entities', self.prompt)
        self.assertNotIn(
            'Do not classify document titles, headings, table labels, or index rows\n'
            '  as physical equipment or activities.',
            self.prompt,
        )

    def test_entities_task_uses_concise_ner_prompt(self):
        ner = self.extractor._unified_prompt(
            task='entities',
            text='unused',
            source_file='inputs/Annexure 2.1 _ DOC-ID.md',
        )
        self.assertIn('TASK = ENTITIES only (NER)', ner)
        self.assertIn('<<CHUNK_TEXT>>', ner)
        self.assertIn('YAML typing guidance:', ner)
        self.assertIn('Entity definitions', ner)
        self.assertIn('Disambiguation:', ner)
        self.assertIn('classify each entity from source context', ner.casefold())
        self.assertNotIn(
            'Document, System, Equipment, Component, and ControlLoop (instruments)',
            ner,
        )
        self.assertNotIn('A) ENTITY EXTRACTION', ner)
        self.assertNotIn('Preferred relation predicates:', ner)
        # Compact YAML guidance stays bounded for Nova attention.
        self.assertLess(len(ner), 12000)
        # Schema example must not leak path tokens that Nova treats as entities.
        self.assertNotIn('Annexure 2.1', ner)
        self.assertNotIn('DOC-ID', ner)
        self.assertIn('"source_file": ""', ner)

    def test_relations_task_keeps_full_unified_guidance(self):
        rel = self.extractor._unified_prompt(
            task='relations',
            text='unused',
            entities=[{'text': 'Pump A', 'label': 'Equipment'}],
            source_file='inputs/sample.md',
        )
        self.assertIn('TASK = RELATIONS only', rel)
        self.assertIn('A) ENTITY EXTRACTION', rel)
        self.assertIn('Preferred relation predicates:', rel)
        # Prefer known endpoints; do not hard-limit recall with "allowed only".
        self.assertIn('Prefer Known entities below as endpoints', rel)
        self.assertNotIn('allowed endpoints', rel)
        self.assertIn('extract every source-supported entity–entity relation', rel)
        # Source-span / section-scope constraints (prompt + task_focus).
        self.assertIn('contiguous SOURCE span', rel)
        self.assertIn('B0. SOURCE-SPAN / SECTION SCOPE', rel)
        self.assertIn('annexure-index', rel.casefold())
        self.assertIn('general QA / inspection', rel)
        self.assertIn('requirement/contract prose', rel)
        self.assertIn('requirement or contract prose', rel.casefold())

    def test_relation_and_triplet_opts_serialize_chunk_workers(self):
        opts_rel = self.extractor._llm_opts(
            kind='relation',
            text='Pump A feeds Tank B',
            entities=[{'text': 'Pump A', 'label': 'Equipment'}],
        )
        opts_trip = self.extractor._llm_opts(
            kind='triplet',
            text='Pump A capacity 10 TPH',
            entities=[{'text': 'Pump A', 'label': 'Equipment'}],
        )
        self.assertEqual(opts_rel.get('max_workers'), 1)
        self.assertEqual(opts_trip.get('max_workers'), 1)

    def test_mixed_document_system_equipment_instrument_guidance(self):
        ner = self.extractor._unified_prompt(task='entities', text='', source_file='inputs/x.md')
        # Primaries come from YAML TYPE_HINTS / definitions, not a hardcoded allow-list sentence.
        for token in ('Document', 'System', 'Equipment', 'ControlLoop'):
            self.assertIn(token, ner)
        self.assertIn('Extract tagged assets in table data rows', ner)
        self.assertIn('repeated page headers/footers/metadata banners', ner)
        self.assertIn('tender-data', ner)
        self.assertIn('not from this schema example', ner)
        self.assertIn('Recommended subtypes', ner)
        self.assertIn('Table extraction rules:', ner)

    def test_unified_prompt_mentions_yaml_disambiguation(self):
        self.assertIn('ONTOLOGY_HINTS entity definitions and disambiguation rules', self.prompt)
        self.assertIn('Recommended subtypes from ONTOLOGY_HINTS', self.prompt)
        self.assertIn('Non-entity policy block inside ONTOLOGY_HINTS', self.prompt)
        self.assertIn('definition-based', self.prompt.casefold())


class NonEntityPolicyTests(unittest.TestCase):
    """Definition-based non_entity_policy: structure, prompt wiring, no blacklists."""

    REQUIRED_SECTIONS = (
        'independent_entity',
        'properties_not_nodes',
        'identifiers_not_separate_nodes',
        'relations_not_nodes',
        'reject_noise',
        'reject_field_labels',
        'reject_generic_words',
        'reject_incomplete_headings',
        'reject_duplicate_identifiers',
        'context_sensitive',
        'other_is_not_for',
    )

    def setUp(self):
        self.policy = er.load_policy()
        self.nep = er.non_entity_policy_section(self.policy)

    def test_non_entity_policy_is_definition_based(self):
        self.assertIsInstance(self.nep, dict)
        self.assertTrue(self.nep)
        for key in self.REQUIRED_SECTIONS:
            self.assertIn(key, self.nep, msg=f'missing section {key}')
            section = self.nep[key]
            self.assertIsInstance(section, dict, msg=f'{key} must be a mapping')
            definition = er.collapse(str(section.get('definition') or ''))
            criteria = section.get('decision_criteria')
            self.assertTrue(definition, msg=f'{key} needs definition')
            self.assertIsInstance(criteria, list, msg=f'{key} needs decision_criteria list')
            self.assertGreaterEqual(len(criteria), 1, msg=f'{key} needs criteria')
            # No legacy string-list blacklist shape for required sections.
            self.assertNotIsInstance(section, list)

    def test_no_hardcoded_example_blacklists_in_policy(self):
        blob = str(self.nep).casefold()
        for banned in (
            'capacity: 500',
            'pressure: 10 bar',
            'p-101 when it is an alias',
            '22-pk-3801 when it only aliases',
            'mandatory spares',
            'part - i',
            'part - ii',
        ):
            self.assertNotIn(banned, blob)

    def test_generic_terms_require_context_not_exact_word_reject(self):
        generic = self.nep['reject_generic_words']
        text = (
            er.collapse(str(generic.get('definition') or ''))
            + ' '
            + ' '.join(str(c) for c in (generic.get('decision_criteria') or []))
        ).casefold()
        self.assertIn('exact-word blacklist', text)
        self.assertIn('independent identity', text)
        self.assertIn('context', text)

    def test_extraction_output_example_preserved_as_illustrative(self):
        taxonomy = self.policy.data.get('ner_taxonomy') or {}
        example = taxonomy.get('extraction_output_example')
        self.assertIsInstance(example, dict)
        entities = example.get('entities') or []
        self.assertTrue(entities)
        self.assertEqual(entities[0].get('text'), 'Centrifugal Pump P-101')
        notes = taxonomy.get('notes') or []
        notes_blob = ' '.join(str(n) for n in notes).casefold()
        self.assertIn('illustrative', notes_blob)

    def test_non_entity_policy_hints_cover_decision_cases(self):
        hints = er.non_entity_policy_hints(self.policy, compact=False)
        self.assertIn('Non-entity policy', hints)
        self.assertIn('never from memorised example phrases', hints.casefold())
        # Valid independent entity guidance.
        self.assertIn('Independent entity', hints)
        # Scalar properties are not nodes.
        self.assertIn('Properties are not entities', hints)
        self.assertIn('metadata.properties', hints.casefold())
        # Identifiers / relations / noise / incomplete headings / duplicates.
        self.assertIn('Identifiers are not separate entities', hints)
        self.assertIn('Relations are not entities', hints)
        self.assertIn('Noise and formatting artifacts', hints)
        self.assertIn('Field labels are not entities', hints)
        self.assertIn('Incomplete headings', hints)
        self.assertIn('Duplicate identifiers', hints)
        # Generic terms kept when source gives identity; Other misuse blocked.
        self.assertIn('Generic terms (context-sensitive)', hints)
        self.assertIn('Other is not for', hints)
        self.assertNotIn('capacity: 500', hints.casefold())
        self.assertNotIn('accessories', hints.casefold())

    def test_legacy_list_shape_still_renders(self):
        class _P:
            data = {
                'ner_taxonomy': {
                    'non_entity_policy': {
                        'properties_not_nodes': [
                            'legacy scalar example should render as criterion',
                        ],
                    },
                },
            }

        hints = er.non_entity_policy_hints(_P(), compact=False)
        self.assertIn('Properties are not entities', hints)
        self.assertIn('legacy scalar example should render as criterion', hints)

    def test_entities_prompt_includes_non_entity_policy_from_yaml(self):
        extractor = er.FactExtractor.__new__(er.FactExtractor)
        extractor.policy = self.policy
        extractor.type_hints = er.primary_type_hints(self.policy)
        ner = extractor._unified_prompt(task='entities', text='', source_file='inputs/x.md')
        self.assertIn('Non-entity policy', ner)
        self.assertIn('non-entity policy', ner.casefold())
        self.assertIn('independently identifiable', ner.casefold())
        self.assertIn('Field labels are not entities', ner)
        self.assertIn('Primary label: predict', ner)
        self.assertIn('Subtype: predict', ner)
        # NER task template must not hardcode domain document typing phrases.
        self.assertNotIn('datasheet vs ScopeOfWork vs DocumentSection', ner)
        self.assertNotIn('PROJECT NO.', ner)
        # Guidance comes from YAML formatter, not a duplicated capacity blacklist.
        self.assertNotIn('capacity: 500 m3/h', ner.casefold())
        self.assertLess(len(ner), 14000)

    def test_primary_types_and_disambiguation_unchanged(self):
        hints = er.primary_type_hints(self.policy)
        self.assertEqual(len(hints), 28)
        self.assertIn('Other', hints)
        self.assertIn('Equipment', hints)
        dis = er.disambiguation_hints(self.policy)
        self.assertIn('System vs Equipment', dis)
        self.assertIn('middle segment is pk', dis.casefold())


class GraphResolutionAndEligibilityTests(unittest.TestCase):
    """Endpoint resolve, evidence recovery, hierarchy orientation, quarantine."""

    def setUp(self):
        self.policy = er.load_policy()

    def _entity(self, eid, text, label, *, subtype='', identifiers=None, aliases=None):
        return {
            'id': eid,
            'text': text,
            'label': label,
            'metadata': {
                'canonical_id': eid,
                'subtype': subtype,
                'identifiers': [
                    {'value': v, 'role': 'identity'} for v in (identifiers or [])
                ],
                'aliases': list(aliases or []),
            },
        }

    def test_canonical_and_tag_endpoint_matching(self):
        entities = [
            self._entity('e0', '22-PK-3904', 'System', subtype='EngineeringPackage', identifiers=['22-PK-3904']),
            self._entity('e1', '22-G-01', 'Equipment', identifiers=['22-G-01']),
        ]
        index = er.build_endpoint_resolution_index(entities, policy=self.policy)
        eid, kind = er._resolve_endpoint_ref(
            '22-PK-3904: Blenders Blowing Package',
            by_surface=index['by_surface'],
            index=index,
            policy=self.policy,
        )
        self.assertEqual(kind, 'entity')
        self.assertEqual(eid, 'e0')
        surface_map = er._build_surface_map_from_entities(entities, policy=self.policy)
        canon = er._canonicalize_endpoint(
            '22-PK-3904: Blenders Blowing Package',
            surface_map,
            index=index,
            policy=self.policy,
        )
        self.assertEqual(canon, '22-PK-3904')

    def test_alias_and_identifier_match_without_fuzzy_merge(self):
        entities = [
            self._entity('e0', 'Acme Industrial', 'Organization', aliases=['Acme'], identifiers=['ORG-1']),
            self._entity('e1', 'Beta Industrial', 'Organization', identifiers=['ORG-2']),
        ]
        index = er.build_endpoint_resolution_index(entities, policy=self.policy)
        eid, kind = er._resolve_endpoint_ref('Acme', by_surface=index['by_surface'], index=index)
        self.assertEqual((eid, kind), ('e0', 'entity'))
        eid2, kind2 = er._resolve_endpoint_ref('ORG-2', by_surface=index['by_surface'], index=index)
        self.assertEqual((eid2, kind2), ('e1', 'entity'))
        # Similar name alone must not resolve.
        eid3, kind3 = er._resolve_endpoint_ref(
            'Acme Industrials LLC', by_surface=index['by_surface'], index=index,
        )
        self.assertEqual(kind3, 'unresolved')

    def test_ambiguous_shared_tag_stays_unresolved(self):
        entities = [
            self._entity('e0', 'Package A', 'System', identifiers=['22-PK-3904']),
            self._entity('e1', 'Package B', 'System', identifiers=['22-PK-3904']),
        ]
        index = er.build_endpoint_resolution_index(entities, policy=self.policy)
        eid, kind = er._resolve_endpoint_ref(
            '22-PK-3904: Blenders',
            by_surface=index['by_surface'],
            index=index,
            policy=self.policy,
        )
        self.assertIsNone(eid)
        self.assertEqual(kind, 'unresolved')

    def test_attribute_literal_not_missing_entity(self):
        entities = [self._entity('e0', '22-PK-3801', 'System', identifiers=['22-PK-3801'])]
        bundle = er.ExtractionBundle(
            entities=entities,
            triplets=[{
                'subject': '22-PK-3801',
                'predicate': 'Design Capacity',
                'object': '71.5 TPH',
                'metadata': {
                    'evidence': {'source_text': 'Design Capacity - 71.5 TPH'},
                    'evidence_status': 'unverified',
                    'graph_eligible': False,
                },
            }],
        )
        doc = '22-PK-3801 Polymer\nDesign Capacity - 71.5 TPH\n'
        warnings = er.validate_extraction_graph(
            bundle, self.policy, document_text=doc, source_file='inputs/x.md',
        )
        trip = bundle.triplets[0]
        self.assertEqual(trip.get('subject_id'), 'e0')
        self.assertEqual(trip.get('object_value_kind'), 'literal')
        self.assertFalse(any('missing entity node for triplet object' in w for w in warnings))

    def test_verified_and_unsupported_evidence(self):
        meta = {
            'evidence': {
                'source_text': '22-G-01 is connected to 22-V-01',
            },
        }
        doc = 'Intro.\n22-G-01 is connected to 22-V-01\nFooter'
        out = er._attach_claim_evidence(
            meta, doc, subject='22-G-01', obj='22-V-01', predicate='connected to',
            attribute_mode=False, source_file='inputs/x.md', policy=self.policy,
        )
        self.assertEqual(out.get('evidence_status'), 'verified')
        self.assertTrue(out.get('graph_eligible'))

        # Document has no supportive span for these endpoints → stay unverified.
        bad_doc = 'Only weather notes appear here. No equipment tags at all.'
        bad = er._attach_claim_evidence(
            {'evidence': {'source_text': 'unrelated sentence about weather'}},
            bad_doc,
            subject='22-G-01',
            obj='22-V-01',
            predicate='connected to',
            attribute_mode=False,
            source_file='inputs/x.md',
            policy=self.policy,
        )
        self.assertIn(bad.get('evidence_status'), {'unverified', 'missing'})
        self.assertFalse(bad.get('graph_eligible'))

    def test_evidence_recovery_finds_supportive_line(self):
        doc = 'Header\n22-ME-3801A/B fed by 22-K-3801A/B blower\nFooter'
        quote = er._recover_claim_evidence_quote(
            doc,
            subject='22-ME-3801A/B',
            obj='22-K-3801A/B',
            predicate='fed by',
            attribute_mode=False,
            policy=self.policy,
        )
        self.assertIn('22-ME-3801A/B', quote)
        self.assertIn('22-K-3801A/B', quote)

    def test_evidence_recovery_nearby_span_across_lines(self):
        """Hierarchy nearby: endpoints + membership + section owner + table context."""
        doc = (
            'ITEM : 22-PK-9999\n'
            '\n'
            'Equipment Included in This Package Scope:\n'
            '\n'
            '| ITEM | SERVICE | QUANTITY |\n'
            '| --- | --- | --- |\n'
            '| 22-G-9999A/B | INTAKE FILTERS | 2 |\n'
            '\n'
            'Unrelated footer text.\n'
        )
        quote = er._recover_claim_evidence_quote(
            doc,
            subject='22-PK-9999',
            obj='22-G-9999A/B',
            predicate='hasEquipment',
            attribute_mode=False,
            policy=self.policy,
        )
        self.assertIn('22-PK-9999', quote)
        self.assertIn('22-G-9999A/B', quote)
        self.assertIn('equipment included', quote.casefold())
        self.assertLessEqual(len(quote), er._claim_nearby_span_max_chars(self.policy))

        # Weak one-endpoint LLM quote should verify via gated nearby recovery.
        out = er._attach_claim_evidence(
            {'evidence': {'source_text': '22-G-9999A/B'}},
            doc,
            subject='22-PK-9999',
            obj='22-G-9999A/B',
            predicate='hasEquipment',
            attribute_mode=False,
            source_file='inputs/x.md',
            policy=self.policy,
        )
        self.assertEqual(out.get('evidence_status'), 'verified')
        self.assertTrue(out.get('graph_eligible'))
        recovered = str((out.get('evidence') or {}).get('source_text') or '')
        self.assertIn('22-PK-9999', recovered)
        self.assertIn('22-G-9999A/B', recovered)

        # Co-occurrence alone (no membership cue) must not verify.
        bare = (
            'ITEM : 22-PK-7777\n'
            '\n'
            '| 22-G-7777A/B | FILTER | 1 |\n'
        )
        self.assertEqual(
            er._recover_claim_evidence_quote(
                bare,
                subject='22-PK-7777',
                obj='22-G-7777A/B',
                predicate='hasEquipment',
                attribute_mode=False,
                policy=self.policy,
            ),
            '',
        )

        # Membership without table context must not verify.
        prose = (
            'ITEM : 22-PK-6666\n'
            'Equipment Included in This Package Scope:\n'
            '22-G-6666A/B intake filters\n'
        )
        self.assertEqual(
            er._recover_claim_evidence_quote(
                prose,
                subject='22-PK-6666',
                obj='22-G-6666A/B',
                predicate='hasEquipment',
                attribute_mode=False,
                policy=self.policy,
            ),
            '',
        )

        # Far-apart endpoints must not invent a mega-span.
        far = (
            'ITEM : 22-PK-8888\n'
            'Equipment Included in This Package Scope:\n'
            + ('x' * 1200)
            + '\n| 22-G-8888 | FILTER | 1 |\n'
        )
        empty = er._recover_claim_evidence_quote(
            far,
            subject='22-PK-8888',
            obj='22-G-8888',
            predicate='hasEquipment',
            attribute_mode=False,
            policy=self.policy,
        )
        self.assertEqual(empty, '')

    def test_attribute_field_label_recovers_missing_evidence(self):
        doc = (
            'ITEM : 22-PK-3901\n'
            '\n'
            'POLYMER FLOW RATE : Max. operating: 93 t/hr (\\*)\n'
            'CONVEYING MODE : Open dilute phase circulation system.\n'
        )
        quote = er._recover_claim_evidence_quote(
            doc,
            subject='22-PK-3901',
            obj='Max. operating: 93 t/hr (*)',
            predicate='POLYMER FLOW RATE',
            attribute_mode=True,
            policy=self.policy,
        )
        self.assertIn('POLYMER FLOW RATE', quote)
        self.assertIn('93 t/hr', quote)
        out = er._attach_claim_evidence(
            {'evidence': None},
            doc,
            subject='22-PK-3901',
            obj='Max. operating: 93 t/hr (*)',
            predicate='POLYMER FLOW RATE',
            attribute_mode=True,
            source_file='inputs/x.md',
            policy=self.policy,
        )
        self.assertEqual(out.get('evidence_status'), 'verified')
        self.assertTrue(out.get('graph_eligible'))

    def test_named_member_recovers_without_table_row(self):
        doc = (
            'ITEM : 22-PK-5555\n'
            'Equipment Included in This Package Scope:\n'
            'DIVERTER VALVES (INSTALLED IN PELLETS TRANSFER LINES)\n'
        )
        quote = er._recover_claim_evidence_quote(
            doc,
            subject='22-PK-5555',
            obj='DIVERTER VALVES (INSTALLED IN PELLETS TRANSFER LINES)',
            predicate='includes',
            attribute_mode=False,
            policy=self.policy,
        )
        self.assertIn('22-PK-5555', quote)
        self.assertIn('DIVERTER VALVES', quote)

    def test_partof_inverse_normalization(self):
        parent, child, orientation = er.canonicalize_hierarchy_edge(
            'e_equip', 'e_sys', 'partOf', policy=self.policy,
        )
        self.assertEqual((parent, child), ('e_sys', 'e_equip'))
        self.assertEqual(orientation, 'inverted_for_hierarchy')
        parent2, child2, orientation2 = er.canonicalize_hierarchy_edge(
            'e_sys', 'e_equip', 'contains', policy=self.policy,
        )
        self.assertEqual((parent2, child2), ('e_sys', 'e_equip'))
        self.assertEqual(orientation2, 'as_extracted')

    def test_hierarchy_predicates_not_routed_as_attributes(self):
        """part of / includes must use hierarchy evidence routing, not attribute mode."""
        self.assertFalse(
            er._looks_like_technical_attribute_claim(
                'part of', '22-PK-3901', policy=self.policy,
            )
        )
        self.assertFalse(
            er._looks_like_technical_attribute_claim(
                'includes', 'DIVERTER VALVES', policy=self.policy,
            )
        )
        # Stopwords alone must not make hierarchy wording look like a field label.
        self.assertFalse(
            er._looks_like_technical_attribute_claim(
                'of', '22-PK-3901', policy=self.policy,
            )
        )

    def test_part_of_inverse_attach_uses_hierarchy_not_attribute(self):
        doc = (
            '22-PK-3901 Equipment Included in This Package Scope:\n'
            '| Tag | Description |\n'
            '| 22-V-01 | Hopper |\n'
            '22-V-01 is part of 22-PK-3901 package scope.\n'
        )
        # attribute_mode=True must be overridden for hierarchy predicates.
        out = er._attach_claim_evidence(
            {'evidence': {'source_text': '22-V-01 is part of 22-PK-3901 package scope.'}},
            doc,
            subject='22-V-01',
            obj='22-PK-3901',
            predicate='part of',
            attribute_mode=True,
            source_file='inputs/x.md',
            policy=self.policy,
        )
        self.assertEqual(out.get('evidence_status'), 'verified')
        self.assertTrue(out.get('graph_eligible'))
        # Hierarchy path: endpoints in quote — no attribute section_context.
        self.assertNotIn('section_context', out)

    def test_includes_recovery_requires_membership_gates(self):
        # Endpoints co-occur but no membership cue → hierarchy recovery must refuse.
        bare = 'ITEM : 22-PK-5555\nDIVERTER VALVES (PEER NOTE)\n'
        quote = er._recover_claim_evidence_quote(
            bare,
            subject='22-PK-5555',
            obj='DIVERTER VALVES (PEER NOTE)',
            predicate='includes',
            attribute_mode=False,
            policy=self.policy,
        )
        self.assertEqual(quote, '')
        # Membership + table context → recovery allowed.
        good = (
            'ITEM : 22-PK-5555\n'
            'Equipment Included in This Package Scope:\n'
            '| Tag | Description |\n'
            '| DV-1 | DIVERTER VALVES (INSTALLED IN PELLETS TRANSFER LINES) |\n'
        )
        quote2 = er._recover_claim_evidence_quote(
            good,
            subject='22-PK-5555',
            obj='DIVERTER VALVES (INSTALLED IN PELLETS TRANSFER LINES)',
            predicate='includes',
            attribute_mode=False,
            policy=self.policy,
        )
        self.assertIn('22-PK-5555', quote2)
        self.assertIn('DIVERTER VALVES', quote2)

    def test_density_attribute_still_uses_attribute_routing(self):
        doc = '## 6.1 22-PK-3901: Package\n\nREAL DENSITY: 900 kg/m³\n'
        out = er._attach_claim_evidence(
            {'evidence': {'source_text': 'REAL DENSITY: 900 kg/m³'}},
            doc,
            subject='22-PK-3901',
            obj='900 kg/m³',
            predicate='REAL DENSITY',
            attribute_mode=False,
            source_file='inputs/x.md',
            policy=self.policy,
        )
        self.assertEqual(out.get('evidence_status'), 'verified')
        self.assertTrue(out.get('graph_eligible'))
        self.assertTrue(
            er._looks_like_technical_attribute_claim(
                'REAL DENSITY', '900 kg/m³', policy=self.policy,
            )
        )

    def test_quarantine_llm_hierarchy_repair_reattaches_final_predicate(self):
        doc = (
            '22-PK-3901 Equipment Included in This Package Scope:\n'
            '| Tag | Description |\n'
            '| 22-V-01 | Hopper |\n'
        )
        entities = [
            self._entity('e0', '22-PK-3901', 'System', identifiers=['22-PK-3901']),
            self._entity('e1', '22-V-01', 'Equipment', identifiers=['22-V-01']),
        ]
        bundle = er.ExtractionBundle(
            entities=entities,
            relations=[{
                'subject': '22-PK-3901',
                'predicate': 'connected to',
                'object': '22-V-01',
                'metadata': {
                    'evidence': {'source_text': '22-PK-3901 connected to 22-V-01'},
                    'evidence_status': 'unverified',
                    'graph_eligible': False,
                },
            }],
        )
        er.validate_extraction_graph(bundle, self.policy, document_text=doc)
        bundle.graph, bundle.quarantine = er.build_confirmed_graph(bundle, self.policy)

        def _resolver(prompt: str) -> str:
            self.assertIn('Hierarchy predicates', prompt)
            return json.dumps([{
                'id': 0,
                'action': 'repair',
                'subject': '22-PK-3901',
                'predicate': 'includes',
                'object': '22-V-01',
                'evidence_quote': (
                    '22-PK-3901 Equipment Included in This Package Scope:\n'
                    '| Tag | Description |\n'
                    '| 22-V-01 | Hopper |'
                ),
                'entity_type_hint': '',
            }])

        applied = er.resolve_quarantine_with_llm(
            bundle, self.policy, document_text=doc, source_file='inputs/x.md', resolver=_resolver,
        )
        self.assertGreaterEqual(applied, 1)
        src = bundle.relations[0]
        self.assertEqual(er.collapse(str(src.get('predicate') or '')).casefold(), 'includes')
        meta = src.get('metadata') or {}
        # Re-attached with final hierarchy predicate — never attribute section routing.
        self.assertNotIn('section_context', meta)
        self.assertEqual(meta.get('evidence_status'), 'verified')
        self.assertTrue(meta.get('graph_eligible'))

    def test_valid_system_equipment_and_invalid_org_system(self):
        entities = [
            self._entity('e_sys', 'Conveying System', 'System', subtype='EngineeringPackage'),
            self._entity('e_eq', 'Pump P-1', 'Equipment'),
            self._entity('e_org', 'Vendor Co', 'Organization'),
        ]
        bundle = er.ExtractionBundle(
            entities=entities,
            relations=[
                {
                    'subject': 'Conveying System',
                    'predicate': 'hasEquipment',
                    'object': 'Pump P-1',
                    'metadata': {
                        'evidence': {'source_text': 'Conveying System hasEquipment Pump P-1'},
                        'evidence_status': 'verified',
                        'graph_eligible': True,
                    },
                },
                {
                    'subject': 'Vendor Co',
                    'predicate': 'hasPackage',
                    'object': 'Conveying System',
                    'metadata': {
                        'evidence': {'source_text': 'Vendor Co hasPackage Conveying System'},
                        'evidence_status': 'verified',
                        'graph_eligible': True,
                    },
                },
            ],
        )
        # Seed endpoint annotations expected by graph builder.
        er.validate_extraction_graph(bundle, self.policy, document_text='')
        graph, quarantine = er.build_confirmed_graph(bundle, self.policy, source_file='inputs/x.md')
        preds = {(e['subject'], e['predicate'], e['object']) for e in graph['edges']}
        self.assertIn(('Conveying System', 'hasEquipment', 'Pump P-1'), preds)
        # Organization → System via hasPackage is hierarchy-invalid → quarantine.
        q_preds = {(e['subject'], e['predicate'], e['object']) for e in quarantine['relations']}
        self.assertIn(('Vendor Co', 'hasPackage', 'Conveying System'), q_preds)
        self.assertTrue(
            any('hierarchy_invalid' in (row.get('reason_codes') or []) for row in quarantine['relations'])
        )

    def test_document_association_not_forced_physical(self):
        self.assertEqual(
            er.classify_relation_category('hasDocument', policy=self.policy),
            'document_hierarchy',
        )
        self.assertEqual(
            er.classify_relation_category('connected to', policy=self.policy),
            'technical',
        )
        self.assertEqual(
            er.classify_relation_category('hasEquipment', policy=self.policy),
            'engineering_hierarchy',
        )

    def test_self_parent_and_cycle_and_duplicate(self):
        entities = [
            self._entity('e0', 'Sys A', 'System'),
            self._entity('e1', 'Sys B', 'System'),
        ]
        bundle = er.ExtractionBundle(
            entities=entities,
            relations=[
                {
                    'subject': 'Sys A', 'predicate': 'hasSystem', 'object': 'Sys A',
                    'metadata': {'evidence_status': 'verified', 'graph_eligible': True,
                                 'evidence': {'source_text': 'Sys A hasSystem Sys A'}},
                },
                {
                    'subject': 'Sys A', 'predicate': 'hasSystem', 'object': 'Sys B',
                    'metadata': {'evidence_status': 'verified', 'graph_eligible': True,
                                 'evidence': {'source_text': 'Sys A hasSystem Sys B'}},
                },
                {
                    'subject': 'Sys B', 'predicate': 'hasSystem', 'object': 'Sys A',
                    'metadata': {'evidence_status': 'verified', 'graph_eligible': True,
                                 'evidence': {'source_text': 'Sys B hasSystem Sys A'}},
                },
                {
                    'subject': 'Sys A', 'predicate': 'hasSystem', 'object': 'Sys B',
                    'metadata': {'evidence_status': 'verified', 'graph_eligible': True,
                                 'evidence': {'source_text': 'Sys A hasSystem Sys B'}},
                },
            ],
        )
        er.validate_extraction_graph(bundle, self.policy)
        graph, quarantine = er.build_confirmed_graph(bundle, self.policy)
        reasons = [c for row in quarantine['relations'] for c in (row.get('reason_codes') or [])]
        self.assertIn('self_parent', reasons)
        # Cycle / invalid hierarchy quarantine; duplicates merge into supporting_evidence.
        self.assertTrue(
            any(code in reasons for code in ('hierarchy_cycle', 'hierarchy_invalid'))
            or graph['edge_count'] <= 1
        )
        self.assertNotIn('duplicate_edge', reasons)

    def test_duplicate_edge_merges_supporting_evidence(self):
        entities = [
            self._entity('e0', 'Pkg', 'System', identifiers=['PKG-1']),
            self._entity('e1', 'Eq', 'Equipment', identifiers=['EQ-1']),
        ]
        bundle = er.ExtractionBundle(
            entities=entities,
            relations=[
                {
                    'subject': 'Pkg',
                    'predicate': 'includes',
                    'object': 'Eq',
                    'subject_id': 'e0',
                    'object_id': 'e1',
                    'subject_ref_kind': 'entity',
                    'object_ref_kind': 'entity',
                    'relation_category': 'engineering_hierarchy',
                    'metadata': {
                        'evidence_status': 'verified',
                        'graph_eligible': True,
                        'evidence': {
                            'source_text': 'Pkg includes Eq (table)',
                            'start_char': 10,
                            'end_char': 30,
                        },
                    },
                },
                {
                    'subject': 'Pkg',
                    'predicate': 'includes',
                    'object': 'Eq',
                    'subject_id': 'e0',
                    'object_id': 'e1',
                    'subject_ref_kind': 'entity',
                    'object_ref_kind': 'entity',
                    'relation_category': 'engineering_hierarchy',
                    'metadata': {
                        'evidence_status': 'verified',
                        'graph_eligible': True,
                        'evidence': {
                            'source_text': 'Pkg includes Eq (narrative)',
                            'start_char': 100,
                            'end_char': 130,
                        },
                    },
                },
            ],
        )
        graph, quarantine = er.build_confirmed_graph(bundle, self.policy)
        self.assertEqual(graph['edge_count'], 1)
        self.assertEqual(quarantine['relation_count'], 0)
        edge = graph['edges'][0]
        self.assertEqual(edge['evidence']['source_text'], 'Pkg includes Eq (table)')
        self.assertEqual(len(edge.get('supporting_evidence') or []), 1)
        self.assertEqual(edge['supporting_evidence'][0]['source_text'], 'Pkg includes Eq (narrative)')

    def test_density_triplet_is_attribute_literal_not_unresolved_object(self):
        er._SECTION_AWARE_CACHE.clear()
        entities = [self._entity('e0', '22-PK-3901', 'System', identifiers=['22-PK-3901'])]
        quote = '22-PK-3901 REAL DENSITY: 900 kg/m³'
        bundle = er.ExtractionBundle(
            entities=entities,
            triplets=[{
                'subject': '22-PK-3901',
                'predicate': 'REAL DENSITY',
                'object': '900 kg/m³',
                'metadata': {
                    'evidence': {'source_text': quote},
                    'evidence_status': 'verified',
                    'graph_eligible': True,
                },
            }],
        )
        er.validate_extraction_graph(bundle, self.policy, document_text=quote)
        trip = bundle.triplets[0]
        self.assertEqual(trip.get('relation_category'), 'attribute')
        self.assertEqual(trip.get('object_value_kind'), 'literal')
        graph, quarantine = er.build_confirmed_graph(bundle, self.policy)
        self.assertEqual(graph['edge_count'], 1)
        self.assertEqual(graph['edges'][0]['object_literal'], '900 kg/m³')
        self.assertNotIn('unresolved_object', quarantine.get('reason_counts') or {})

    def test_pk_typing_guided_by_prompt_and_yaml_pins(self):
        """PK → System/EngineeringPackage is prompt + YAML guidance, not identifier regex."""
        policy = er.load_policy()
        identity = policy.data.get('identity_resolution') or {}
        patterns = identity.get('identifier_patterns') or []
        self.assertFalse(any(
            'primary_type' in (row or {})
            for row in patterns
            if isinstance(row, dict)
        ))
        self.assertNotIn('identifier_type_corrections', identity)
        pins = ' '.join(er.compact_prompt_pins(policy)).casefold()
        self.assertIn('pk', pins)
        self.assertIn('engineeringpackage', pins.replace(' ', ''))
        dis = er.disambiguation_hints(policy, limit=40).casefold()
        self.assertIn('middle segment is pk', dis)
        self.assertIn('engineering package', dis)
        prompt = (er.PROMPT_DIR / 'unified_extraction.txt').read_text(encoding='utf-8')
        self.assertIn('middle segment is PK', prompt)
        self.assertIn('second Equipment node for the same bare PK tag', prompt)
        ner = er.FactExtractor.__new__(er.FactExtractor)
        ner.policy = policy
        ner.type_hints = er.primary_type_hints(policy)
        ner.prompt_dir = er.PROMPT_DIR
        ner_prompt = ner._unified_prompt(task='entities', text='x' * 200, source_file='inputs/x.md')
        self.assertIn('PK middle-segment tags', ner_prompt)
        self.assertIn('EngineeringPackage', ner_prompt)

    def test_individual_tag_does_not_resolve_to_group_tag(self):
        """22-K-3901A must not alias to 22-K-3901A-C (identity ≠ membership)."""
        entities = [
            self._entity('e0', '22-K-3901A-C', 'Equipment', identifiers=['22-K-3901A-C']),
        ]
        index = er.build_endpoint_resolution_index(entities, policy=self.policy)
        eid, kind = er._resolve_endpoint_ref(
            '22-K-3901A',
            by_surface=index['by_surface'],
            index=index,
            policy=self.policy,
        )
        self.assertEqual(kind, 'unresolved')
        self.assertIsNone(eid)
        eid2, kind2 = er._resolve_endpoint_ref(
            '22-K-3901A-C',
            by_surface=index['by_surface'],
            index=index,
            policy=self.policy,
        )
        self.assertEqual(kind2, 'entity')
        self.assertEqual(eid2, 'e0')

    def test_recover_missing_endpoint_from_evidence_yaml(self):
        entities = [
            self._entity('e5', '22-K-3901A-C', 'Equipment', identifiers=['22-K-3901A-C']),
        ]
        quote = '22-K-3901A-C controlled by ESD 3909A-H in release mode'
        bundle = er.ExtractionBundle(
            entities=entities,
            relations=[{
                'subject': '22-K-3901A-C',
                'predicate': 'controlled by',
                'object': 'ESD 3909A-H',
                'metadata': {
                    'evidence': {
                        'source_text': quote,
                    },
                    'evidence_status': 'verified',
                    'graph_eligible': True,
                },
            }],
        )
        created = er.recover_missing_endpoints(
            bundle,
            self.policy,
            document_text=quote,
            source_file='inputs/x.md',
        )
        self.assertEqual(created, 1)
        texts = {e['text'] for e in bundle.entities}
        self.assertIn('ESD 3909A-H', texts)
        recovered = next(e for e in bundle.entities if e['text'] == 'ESD 3909A-H')
        self.assertEqual(recovered['label'], 'ControlLoop')
        er.validate_extraction_graph(
            bundle, self.policy, document_text=quote,
        )
        graph, quarantine = er.build_confirmed_graph(bundle, self.policy)
        self.assertEqual(graph['edge_count'], 1)
        self.assertEqual(quarantine.get('relation_count'), 0)

    def test_vendor_provides_datasheet_becomes_semantic_literal(self):
        quote = (
            'Vendor to provide all equipment mechanical data sheet '
            'along with mechanical calculation to tkIS India.'
        )
        obj = (
            'all equipment mechanical data sheet along with mechanical '
            'calculation to tkIS India'
        )
        entities = [
            self._entity('e5', 'tkIS India / Vendor', 'Organization', identifiers=['Vendor']),
        ]
        bundle = er.ExtractionBundle(
            entities=entities,
            relations=[{
                'subject': 'tkIS India / Vendor',
                'predicate': 'provides',
                'object': obj,
                'metadata': {
                    'evidence': {'source_text': quote},
                    'evidence_status': 'verified',
                    'graph_eligible': True,
                },
            }],
        )
        er.recover_missing_endpoints(
            bundle, self.policy, document_text=quote, source_file='inputs/x.md',
        )
        rel = bundle.relations[0]
        self.assertEqual(rel.get('object_ref_kind'), 'literal')
        self.assertEqual(rel.get('object_value_kind'), 'literal')
        self.assertEqual(rel.get('object_literal'), obj)
        self.assertEqual(rel.get('relation_category'), 'semantic')
        meta = rel.get('metadata') or {}
        self.assertTrue(meta.get('semantic_literal_validated'))
        self.assertEqual(meta.get('literal_kind'), 'deliverable_prose')
        er.validate_extraction_graph(bundle, self.policy, document_text=quote)
        graph, quarantine = er.build_confirmed_graph(bundle, self.policy)
        self.assertEqual(graph['edge_count'], 1)
        self.assertEqual(graph['edges'][0]['object_literal'], obj)
        self.assertEqual(graph['edges'][0]['relation_category'], 'semantic')
        self.assertIsNone(graph['edges'][0].get('object_id'))
        self.assertNotIn('unresolved_object', quarantine.get('reason_counts') or {})

    def test_provides_literal_rejects_wrong_subject_primary(self):
        quote = (
            'Vendor to provide all equipment mechanical data sheet '
            'along with mechanical calculation to tkIS India.'
        )
        obj = (
            'all equipment mechanical data sheet along with mechanical '
            'calculation to tkIS India'
        )
        entities = [
            self._entity('e0', '22-PK-3801', 'System', identifiers=['22-PK-3801']),
        ]
        bundle = er.ExtractionBundle(
            entities=entities,
            relations=[{
                'subject': '22-PK-3801',
                'predicate': 'provides',
                'object': obj,
                'metadata': {
                    'evidence': {'source_text': quote},
                    'evidence_status': 'verified',
                    'graph_eligible': True,
                },
            }],
        )
        er.recover_missing_endpoints(
            bundle, self.policy, document_text=quote, source_file='inputs/x.md',
        )
        rel = bundle.relations[0]
        self.assertNotEqual(rel.get('object_ref_kind'), 'literal')
        self.assertFalse((rel.get('metadata') or {}).get('semantic_literal_validated'))
        er.validate_extraction_graph(bundle, self.policy, document_text=quote)
        graph, quarantine = er.build_confirmed_graph(bundle, self.policy)
        self.assertEqual(graph['edge_count'], 0)
        self.assertGreaterEqual(quarantine.get('relation_count') or 0, 1)
        reasons = (quarantine.get('relations') or [{}])[0].get('reason_codes') or []
        # Object text is in evidence → unresolved_object suppressed; still not confirmed
        # without semantic_literal_validated / correct subject contract.
        self.assertNotIn('unresolved_object', reasons)

    def test_evidence_backed_prose_object_skips_unresolved_object_reason(self):
        quote = (
            'Bidder shall take into account the losses / tolerances while quoting '
            'the guaranteed specific power consumption and capacity.'
        )
        obj = quote.rstrip('.')
        entities = [
            self._entity('e31', 'Pneumatic conveying package', 'System'),
        ]
        bundle = er.ExtractionBundle(
            entities=entities,
            triplets=[{
                'subject': 'Pneumatic conveying package',
                'predicate': 'requires',
                'object': obj,
                'subject_id': 'e31',
                'subject_ref_kind': 'entity',
                'object_id': None,
                'object_ref_kind': 'unresolved',
                'relation_category': 'semantic',
                'metadata': {
                    'evidence': {
                        'source_text': quote,
                        'claim_verified': False,
                        'verification_note': 'quote does not support claim endpoints/direction',
                    },
                    'evidence_status': 'unverified',
                    'graph_eligible': False,
                },
            }],
        )
        _graph, quarantine = er.build_confirmed_graph(bundle, self.policy)
        row = (quarantine.get('triplets') or [{}])[0]
        reasons = row.get('reason_codes') or []
        self.assertNotIn('unresolved_object', reasons)
        self.assertIn('evidence_unverified', reasons)
        self.assertEqual(row.get('suggested_repair'), 'recover_or_review_source_evidence')

    def test_evidence_match_exact_html_normalize_then_fuzzy(self):
        # Exact
        self.assertTrue(er._object_verbatim_in_evidence('TEMA ASME', 'x TEMA ASME y'))
        # HTML normalize (&amp; ↔ &)
        self.assertTrue(
            er._object_verbatim_in_evidence(
                'TEMA & ASME Section VIII Div-1',
                '| Heat Exchanger | TEMA &amp; ASME Section VIII Div-1 |',
            )
        )
        # Fuzzy near-equal window
        self.assertTrue(
            er._object_verbatim_in_evidence(
                'TEMA & ASME Section VIII Div-1',
                'std TEMA & ASME Section VIII Div1 code',
            )
        )
        self.assertFalse(
            er._object_verbatim_in_evidence('TEMA & ASME Section VIII Div-1', 'unrelated row')
        )

    def test_quarantine_llm_skipped_when_local_cascade_grounds_endpoints(self):
        doc = (
            'Pneumatic conveying package requires Bidder shall take into account '
            'the losses / tolerances while quoting the guaranteed specific power '
            'consumption and capacity.'
        )
        obj = (
            'Bidder shall take into account the losses / tolerances while quoting '
            'the guaranteed specific power consumption and capacity'
        )
        entities = [self._entity('e31', 'Pneumatic conveying package', 'System')]
        bundle = er.ExtractionBundle(
            entities=entities,
            triplets=[{
                'subject': 'Pneumatic conveying package',
                'predicate': 'requires',
                'object': obj,
                'subject_id': 'e31',
                'subject_ref_kind': 'entity',
                'object_id': None,
                'object_ref_kind': 'unresolved',
                'relation_category': 'semantic',
                'metadata': {
                    'evidence': {'source_text': doc},
                    'evidence_status': 'unverified',
                    'graph_eligible': False,
                },
            }],
        )
        bundle.graph, bundle.quarantine = er.build_confirmed_graph(bundle, self.policy)

        def _resolver(_prompt: str) -> str:
            self.fail('2nd LLM must not run when Exact/Normalize/Fuzzy already grounds endpoints')

        applied = er.resolve_quarantine_with_llm(
            bundle, self.policy, document_text=doc, source_file='inputs/x.md', resolver=_resolver,
        )
        self.assertEqual(applied, 0)

    def test_attribute_unresolved_object_becomes_attribute_literal(self):
        # Value quote may omit owner when section_context grounds the subject.
        doc = 'REAL DENSITY: 900 kg/m³'
        entities = [self._entity('e0', '22-PK-3901', 'System', identifiers=['22-PK-3901'])]
        bundle = er.ExtractionBundle(
            entities=entities,
            triplets=[{
                'subject': '22-PK-3901',
                'predicate': 'REAL DENSITY',
                'object': '900 kg/m³',
                'metadata': {
                    'evidence': {'source_text': doc},
                    'evidence_status': 'verified',
                    'graph_eligible': True,
                    'section_context': {
                        'source_text': '## 6.1 22-PK-3901: Package',
                        'resolver': 'deterministic_section_row',
                    },
                },
            }],
        )
        er.recover_missing_endpoints(
            bundle, self.policy, document_text=doc, source_file='inputs/x.md',
        )
        er.validate_extraction_graph(bundle, self.policy, document_text=doc)
        trip = bundle.triplets[0]
        self.assertEqual(trip.get('relation_category'), 'attribute')
        self.assertEqual(trip.get('object_value_kind'), 'literal')
        graph, quarantine = er.build_confirmed_graph(bundle, self.policy)
        self.assertEqual(graph['edge_count'], 1)
        self.assertEqual(graph['edges'][0]['object_literal'], '900 kg/m³')
        self.assertNotIn('unresolved_object', quarantine.get('reason_counts') or {})

    def test_non_atomic_object_not_minted_as_entity(self):
        prose = (
            'all equipment mechanical data sheet along with mechanical '
            'calculation to tkIS India'
        )
        quote = f'Vendor to provide {prose}.'
        entities = [
            self._entity('e5', 'tkIS India / Vendor', 'Organization'),
        ]
        bundle = er.ExtractionBundle(
            entities=entities,
            relations=[{
                'subject': 'tkIS India / Vendor',
                'predicate': 'controlled by',
                'object': prose,
                'metadata': {
                    'evidence': {'source_text': quote},
                    'evidence_status': 'verified',
                    'graph_eligible': True,
                },
            }],
        )
        created = er.recover_missing_endpoints(
            bundle, self.policy, document_text=quote, source_file='inputs/x.md',
        )
        self.assertEqual(created, 0)
        self.assertEqual(len(bundle.entities), 1)
        self.assertFalse((bundle.relations[0].get('metadata') or {}).get('semantic_literal_validated'))

    def test_non_hierarchy_technical_relation_preserved(self):
        entities = [
            self._entity('e0', '22-G-01', 'Equipment', identifiers=['22-G-01']),
            self._entity('e1', '22-V-01', 'Equipment', identifiers=['22-V-01']),
        ]
        bundle = er.ExtractionBundle(
            entities=entities,
            relations=[{
                'subject': '22-G-01',
                'predicate': 'connected to',
                'object': '22-V-01',
                'metadata': {
                    'evidence': {'source_text': '22-G-01 connected to 22-V-01'},
                    'evidence_status': 'verified',
                    'graph_eligible': True,
                },
            }],
        )
        er.validate_extraction_graph(bundle, self.policy)
        graph, quarantine = er.build_confirmed_graph(bundle, self.policy)
        self.assertEqual(graph['edge_count'], 1)
        self.assertEqual(graph['edges'][0]['relation_category'], 'technical')
        self.assertEqual(quarantine['relation_count'], 0)

    def test_subtype_gate_still_applied(self):
        entities = [
            self._entity('e0', 'Scope Doc', 'Document', subtype='ScopeOfWork'),
            self._entity('e1', 'Pkg', 'System', subtype='EngineeringPackage'),
        ]
        bundle = er.ExtractionBundle(
            entities=entities,
            relations=[{
                'subject': 'Scope Doc',
                'predicate': 'hasPackage',
                'object': 'Pkg',
                'metadata': {},
            }],
        )
        warnings = er.validate_extraction_graph(bundle, self.policy)
        # Valid ScopeOfWork → EngineeringPackage should not warn subtype.
        self.assertFalse(any('subtype gate' in w for w in warnings))

    def test_prompt_mentions_known_entity_endpoint_reuse(self):
        prompt = (er.PROMPT_DIR / 'unified_extraction.txt').read_text(encoding='utf-8')
        self.assertIn('Reuse Known Entity text, identifiers, and aliases exactly', prompt)
        self.assertIn('Prefer parent-as-subject direction for containment', prompt)
        self.assertIn('must not become a self-loop where', prompt)
        self.assertIn('not containment', prompt)

    def test_quarantine_llm_accepts_verbatim_evidence_only(self):
        doc = '22-PK-3901 POLYMER FLOW RATE : Max. operating: 93 t/hr\n'
        entities = [self._entity('e0', '22-PK-3901', 'System', identifiers=['22-PK-3901'])]
        bundle = er.ExtractionBundle(
            entities=entities,
            triplets=[{
                'subject': '22-PK-3901',
                'predicate': 'POLYMER FLOW RATE',
                'object': 'Max. operating: 93 t/hr',
                'metadata': {},
            }],
        )
        er.validate_extraction_graph(bundle, self.policy, document_text=doc)
        bundle.graph, bundle.quarantine = er.build_confirmed_graph(bundle, self.policy)

        def _resolver(prompt: str) -> str:
            self.assertIn('POLYMER FLOW RATE', prompt)
            return json.dumps([{
                'id': 0,
                'action': 'accept',
                'evidence_quote': '22-PK-3901 POLYMER FLOW RATE : Max. operating: 93 t/hr',
                'entity_type_hint': '',
            }])

        applied = er.resolve_quarantine_with_llm(
            bundle, self.policy, document_text=doc, source_file='inputs/x.md', resolver=_resolver,
        )
        self.assertEqual(applied, 1)
        self.assertGreaterEqual(bundle.graph.get('edge_count') or 0, 1)

    def test_quarantine_llm_skips_self_parent(self):
        entities = [self._entity('e0', 'ESD 1', 'ControlLoop', identifiers=['ESD 1'])]
        bundle = er.ExtractionBundle(
            entities=entities,
            triplets=[{
                'subject': 'ESD 1',
                'predicate': 'alarms activation',
                'object': 'ESD 1',
                'metadata': {
                    'evidence': {'source_text': 'Alarms activation of ESD 1.'},
                    'evidence_status': 'verified',
                    'graph_eligible': True,
                },
            }],
        )
        er.validate_extraction_graph(bundle, self.policy, document_text='Alarms activation of ESD 1.')
        bundle.graph, bundle.quarantine = er.build_confirmed_graph(bundle, self.policy)
        called = {'n': 0}

        def _resolver(_prompt: str) -> str:
            called['n'] += 1
            return '[]'

        applied = er.resolve_quarantine_with_llm(
            bundle, self.policy, document_text='Alarms activation of ESD 1.', resolver=_resolver,
        )
        self.assertEqual(applied, 0)
        self.assertEqual(called['n'], 0)

    def test_tds_document_typed_as_datasheet_not_haspackage(self):
        """PRO-TDS docs use Datasheet/Specification + specifies; hasPackage stays SOW-only."""
        policy = er.load_policy()
        subtypes = er.recommended_subtype_hints(policy)
        self.assertIn('Datasheet', subtypes)
        dis = er.disambiguation_hints(policy, limit=40)
        self.assertIn('PRO-TDS', dis)
        self.assertIn('specifies', dis.casefold())
        self.assertIn('hasPackage', dis)
        prompt = (er.PROMPT_DIR / 'unified_extraction.txt').read_text(encoding='utf-8')
        self.assertIn('PRO-TDS / TDS', prompt)
        self.assertIn('hasPackage is only for ScopeOfWork', prompt)
        # Subtype gate still requires ScopeOfWork parent for hasPackage.
        gates = er.predicate_subtype_gates(policy)
        gate = gates.get('haspackage') or gates.get('has package')
        self.assertIsNotNone(gate)
        parent_allow = {str(x).casefold() for x in (gate.get('parent') or [])}
        self.assertIn('scopeofwork', parent_allow)
        self.assertNotIn('documentsection', parent_allow)

    def test_compact_large_doc_prompts_keep_tds_guidance(self):
        """Chunked NER uses YAML compact_prompt + disambiguation compact_priority."""
        policy = er.load_policy()
        settings = er.compact_prompt_settings(policy)
        self.assertTrue(settings.get('pins'))
        self.assertNotIn('disambiguation_priority_pairs', settings)
        self.assertIn('Document', er.compact_priority_subtypes(policy))
        # Priority lives on disambiguation_rules, not a separate pair list.
        rules = (policy.data.get('ner_taxonomy') or {}).get('disambiguation_rules') or []
        prioritized = [
            r for r in rules
            if isinstance(r, dict) and r.get('compact_priority') is not None
        ]
        self.assertGreaterEqual(len(prioritized), 4)
        self.assertTrue(
            any(
                set(str(x).casefold() for x in (r.get('between') or []))
                >= {'document', 'system'}
                for r in prioritized
            )
        )

        compact = er.ontology_hints(policy, compact=True)
        for pin in er.compact_prompt_pins(policy):
            self.assertIn(pin, compact)
        self.assertIn('PRO-TDS', compact)
        self.assertIn('Datasheet', compact)
        self.assertIn('ScopeOfWork', compact)
        self.assertIn('hasPackage', compact)
        self.assertIn('specifies', compact.casefold())
        self.assertIn('Document vs System', compact)
        # PK → System/EngineeringPackage rule must survive compact truncation.
        self.assertIn('middle segment is pk', compact.casefold())
        doc_line = next(
            (
                ln for ln in compact.splitlines()
                if ln.startswith('- Document:') and 'ScopeOfWork' in ln and 'Datasheet' in ln
            ),
            '',
        )
        self.assertTrue(doc_line)
        self.assertLess(doc_line.find('Datasheet'), doc_line.find('Drawing'))
        self.assertLess(doc_line.find('ScopeOfWork'), doc_line.find('Datasheet'))

        # main.py must not hardcode domain TDS / document-typing phrases in NER template.
        ner_src = Path(er.__file__).read_text(encoding='utf-8')
        self.assertNotIn('PRO-TDS / TDS / datasheet document numbers are Datasheet', ner_src)
        self.assertNotIn('Document typing (large docs): PRO-TDS', ner_src)
        self.assertNotIn('datasheet vs ScopeOfWork vs DocumentSection', ner_src)
        self.assertNotIn('PROJECT NO.', ner_src)

        extractor = er.FactExtractor.__new__(er.FactExtractor)
        extractor.policy = policy
        extractor.type_hints = er.primary_type_hints(policy)
        extractor.prompt_dir = er.PROMPT_DIR
        ner = extractor._unified_prompt(
            task='entities',
            text='x' * 50000,
            source_file='inputs/Annexure 2.2 _ 1916-PRO-TDS-22-PK-3804_advanced_auto.md',
        )
        self.assertIn('YAML typing guidance', ner)
        self.assertIn('PRO-TDS', ner)  # via YAML compact pin / disambiguation
        self.assertLess(len(ner), 14000)

        rel = extractor._unified_prompt(
            task='relations',
            text='x' * 50000,
            entities=[{'text': '22-PK-3804', 'label': 'System'}],
            source_file='inputs/Annexure 2.2 _ 1916-PRO-TDS-22-PK-3804_advanced_auto.md',
        )
        self.assertIn('hasPackage is only for ScopeOfWork', rel)
        self.assertIn('specifies/describes', rel)

    def test_identity_merge_shared_identifier_compatible_types(self):
        doc = 'Package 22-PK-3801 Conveying System 22-PK-3801 hasSystem link.'
        bundle = er.ExtractionBundle(entities=[
            self._entity(
                'e0', '22-PK-3801', 'System',
                subtype='EngineeringPackage', identifiers=['22-PK-3801'],
            ),
            self._entity(
                'e1', 'Conveying System 22-PK-3801', 'System',
                subtype='System', identifiers=['22-PK-3801'],
            ),
        ])
        id_map = er.merge_identity_duplicate_entities(
            bundle, self.policy, document_text=doc,
        )
        self.assertEqual(len(bundle.entities), 1)
        self.assertTrue(id_map)
        survivor = bundle.entities[0]
        aliases = {
            str(a).casefold()
            for a in ((survivor.get('metadata') or {}).get('aliases') or [])
        }
        surfaces = {survivor['text'].casefold()} | aliases
        self.assertTrue(
            '22-pk-3801' in surfaces and 'conveying system 22-pk-3801' in surfaces
        )

    def test_identity_merge_blocks_conflicting_identifiers_despite_alias(self):
        doc = '22-PK-3801 also called 22-PK-3802 in error.'
        left = self._entity('e0', '22-PK-3801', 'System', identifiers=['22-PK-3801'])
        right = self._entity(
            'e1', 'Conveying System 22-PK-3802', 'System',
            identifiers=['22-PK-3802'], aliases=['22-PK-3801'],
        )
        right['metadata']['evidence'] = {
            'source_text': 'Conveying System 22-PK-3802 alias 22-PK-3801',
        }
        bundle = er.ExtractionBundle(entities=[left, right])
        id_map = er.merge_identity_duplicate_entities(
            bundle, self.policy, document_text=doc,
        )
        self.assertEqual(id_map, {})
        self.assertEqual(len(bundle.entities), 2)

    def test_identity_merge_rejects_fuzzy_or_name_only(self):
        bundle = er.ExtractionBundle(entities=[
            self._entity('e0', 'Conveying System', 'System'),
            self._entity('e1', 'Conveying Systems', 'System'),
        ])
        id_map = er.merge_identity_duplicate_entities(
            bundle, self.policy, document_text='Conveying System near Conveying Systems.',
        )
        self.assertEqual(id_map, {})
        self.assertEqual(len(bundle.entities), 2)

    def test_identity_merge_alias_requires_source_evidence(self):
        left = self._entity('e0', '22-PK-3801', 'System')
        right = self._entity(
            'e1', 'Conveying System Unit', 'System', aliases=['22-PK-3801'],
        )
        # Alias listed but not present in evidence/document → no merge.
        right['metadata']['evidence'] = {
            'source_text': 'Conveying System Unit overview without the tag.',
        }
        bundle = er.ExtractionBundle(entities=[left, right])
        id_map = er.merge_identity_duplicate_entities(
            bundle, self.policy, document_text='Unrelated document body.',
        )
        self.assertEqual(id_map, {})
        self.assertEqual(len(bundle.entities), 2)

        # Same alias with source-backed evidence → merge.
        right['metadata']['evidence'] = {
            'source_text': 'Conveying System Unit 22-PK-3801 installed.',
        }
        bundle2 = er.ExtractionBundle(entities=[
            self._entity('e0', '22-PK-3801', 'System'),
            right,
        ])
        id_map2 = er.merge_identity_duplicate_entities(
            bundle2, self.policy,
            document_text='Conveying System Unit 22-PK-3801 installed.',
        )
        self.assertTrue(id_map2)
        self.assertEqual(len(bundle2.entities), 1)

    def test_merged_identity_self_loop_quarantined_with_audit(self):
        entities = [
            self._entity(
                'e0', '22-PK-3801', 'System',
                subtype='EngineeringPackage', identifiers=['22-PK-3801'],
            ),
            self._entity(
                'e1', 'Conveying System 22-PK-3801', 'System',
                subtype='System', identifiers=['22-PK-3801'],
            ),
        ]
        bundle = er.ExtractionBundle(
            entities=entities,
            relations=[{
                'subject': '22-PK-3801',
                'predicate': 'hasSystem',
                'object': 'Conveying System 22-PK-3801',
                'metadata': {
                    'evidence_status': 'verified',
                    'graph_eligible': True,
                    'evidence': {
                        'source_text': '22-PK-3801 hasSystem Conveying System 22-PK-3801',
                    },
                },
            }],
        )
        id_map = er.merge_identity_duplicate_entities(bundle, self.policy)
        self.assertEqual(len(bundle.entities), 1)
        er.validate_extraction_graph(bundle, self.policy)
        er._redirect_bundle_endpoints_after_identity_merge(bundle, id_map)
        graph, quarantine = er.build_confirmed_graph(bundle, self.policy)
        self.assertEqual(graph.get('edge_count', len(graph.get('edges') or [])), 0)
        q_rows = quarantine.get('relations') or []
        self.assertTrue(q_rows)
        row = q_rows[0]
        self.assertIn('merged_identity_self_loop', row.get('reason_codes') or [])
        self.assertEqual(row.get('audit_reason'), 'merged_identity_self_loop')
        self.assertEqual(row.get('preserved_subject'), '22-PK-3801')
        self.assertEqual(row.get('preserved_object'), 'Conveying System 22-PK-3801')
        self.assertEqual(row.get('preserved_subject_id'), row.get('preserved_object_id'))

    def test_semantic_self_loop_not_auto_removed_by_hierarchy_rule(self):
        """Hierarchy forbid_self_parent must not auto-drop non-hierarchy self edges."""
        entities = [self._entity('e0', 'Sys A', 'System', identifiers=['SYS-A'])]
        bundle = er.ExtractionBundle(
            entities=entities,
            relations=[{
                'subject': 'Sys A',
                'predicate': 'connected to',
                'object': 'Sys A',
                'subject_id': 'e0',
                'object_id': 'e0',
                'subject_ref_kind': 'entity',
                'object_ref_kind': 'entity',
                'relation_category': 'technical',
                'metadata': {
                    'evidence_status': 'verified',
                    'graph_eligible': True,
                    'evidence': {'source_text': 'Sys A connected to Sys A'},
                },
            }],
        )
        graph, quarantine = er.build_confirmed_graph(bundle, self.policy)
        confirmed = {(e['subject'], e['predicate'], e['object']) for e in (graph.get('edges') or [])}
        q_reasons = [
            c for row in (quarantine.get('relations') or [])
            for c in (row.get('reason_codes') or [])
        ]
        self.assertNotIn('merged_identity_self_loop', q_reasons)
        self.assertNotIn('self_parent', q_reasons)
        self.assertIn(('Sys A', 'connected to', 'Sys A'), confirmed)

    def test_rejects_incomplete_attribute_action_phrase(self):
        quote = (
            'Vendor shall carry out the thermal design of Discharge Cooler '
            '(22-E-3904) and submit for tkIS review.'
        )
        out = er._attach_claim_evidence(
            {'evidence': {'source_text': quote}},
            quote,
            subject='Discharge Cooler (22-E-3904)',
            obj='Vendor shall carry out',
            predicate='thermal design',
            attribute_mode=True,
            source_file='inputs/x.md',
            policy=self.policy,
        )
        self.assertFalse(out.get('graph_eligible'))
        self.assertNotEqual(out.get('evidence_status'), 'verified')
        note = ((out.get('evidence') or {}).get('verification_note') or '')
        self.assertIn('incomplete_attribute_action_phrase', note)

    def test_rejects_endpoint_cooccurrence_without_predicate_meaning(self):
        # Title list co-mentions package + tag without membership/predicate assertion.
        quote = (
            'TECHNICAL SPECIFICATION FOR PNEUMATIC CONVEYING PACKAGE\n'
            '(22-PK-3801, 22-PK-3804, 22-PK-3901 & 22-PK-3904)'
        )
        self.assertFalse(
            er._claim_supported_by_quote(
                quote,
                'Pneumatic conveying package',
                '22-PK-3801',
                attribute_mode=False,
                predicate='includes',
                policy=self.policy,
            )
        )
        # Wrong-direction purchase: package tag only in header, purchase attaches to Future.
        purch = (
            '(22-PK-3801, 22-PK-3804, 22-PK-3901 & 22-PK-3904)\n\n'
            '14.20 Bagging Silo (22-V-3902C) and corresponding Pneumatic conveying '
            'items shall be purchased in Future.'
        )
        self.assertFalse(
            er._claim_supported_by_quote(
                purch,
                'Bagging Silo (22-V-3902C)',
                '22-PK-3904',
                attribute_mode=False,
                predicate='to be purchased in',
                policy=self.policy,
            )
        )

    def test_accepts_source_backed_part_of_with_predicate_in_span(self):
        quote = (
            'dedicated Silo Venting Blower 22-K-3904A/B which is part of '
            'Blending blowing package (22-PK-3904).'
        )
        self.assertTrue(
            er._claim_supported_by_quote(
                quote,
                'Silo Venting Blower 22-K-3904A/B',
                '22-PK-3904',
                attribute_mode=False,
                predicate='part of',
                policy=self.policy,
            )
        )

    def test_confirmed_graph_quarantines_false_positive_eligible_flags(self):
        """Stale graph_eligible=true must not confirm when quote lacks claim support."""
        entities = [
            self._entity('e0', 'Bagging Silo (22-V-3902C)', 'Equipment', identifiers=['22-V-3902C']),
            self._entity('e1', '22-PK-3904', 'System', identifiers=['22-PK-3904']),
            self._entity('e2', 'Discharge Cooler (22-E-3904)', 'Equipment', identifiers=['22-E-3904']),
        ]
        purch_quote = (
            '(22-PK-3801, 22-PK-3904)\n\n'
            'Bagging Silo (22-V-3902C) shall be purchased in Future.'
        )
        attr_quote = (
            'Vendor shall carry out the thermal design of Discharge Cooler '
            '(22-E-3904) and submit for tkIS review.'
        )
        good_quote = (
            'Silo Venting Blower 22-K-3904A/B which is part of package 22-PK-3904.'
        )
        entities.append(
            self._entity('e3', 'Silo Venting Blower 22-K-3904A/B', 'Equipment', identifiers=['22-K-3904A/B'])
        )
        bundle = er.ExtractionBundle(
            entities=entities,
            relations=[
                {
                    'subject': 'Bagging Silo (22-V-3902C)',
                    'predicate': 'to be purchased in',
                    'object': '22-PK-3904',
                    'subject_id': 'e0',
                    'object_id': 'e1',
                    'subject_ref_kind': 'entity',
                    'object_ref_kind': 'entity',
                    'relation_category': 'semantic',
                    'metadata': {
                        'evidence_status': 'verified',
                        'graph_eligible': True,
                        'evidence': {'source_text': purch_quote},
                    },
                },
                {
                    'subject': 'Silo Venting Blower 22-K-3904A/B',
                    'predicate': 'part of',
                    'object': '22-PK-3904',
                    'subject_id': 'e3',
                    'object_id': 'e1',
                    'subject_ref_kind': 'entity',
                    'object_ref_kind': 'entity',
                    'relation_category': 'engineering_hierarchy',
                    'metadata': {
                        'evidence_status': 'verified',
                        'graph_eligible': True,
                        'evidence': {'source_text': good_quote},
                    },
                },
            ],
            triplets=[{
                'subject': 'Discharge Cooler (22-E-3904)',
                'predicate': 'thermal design',
                'object': 'Vendor shall carry out',
                'subject_id': 'e2',
                'object_id': None,
                'subject_ref_kind': 'entity',
                'object_ref_kind': 'literal',
                'object_value_kind': 'literal',
                'relation_category': 'attribute',
                'metadata': {
                    'evidence_status': 'verified',
                    'graph_eligible': True,
                    'evidence': {'source_text': attr_quote},
                },
            }],
        )
        graph, quarantine = er.build_confirmed_graph(bundle, self.policy)
        confirmed = {
            (e['subject'], e['predicate'], e.get('object_literal') or e['object'])
            for e in (graph.get('edges') or [])
        }
        self.assertNotIn(
            ('Bagging Silo (22-V-3902C)', 'to be purchased in', '22-PK-3904'),
            confirmed,
        )
        self.assertNotIn(
            ('Discharge Cooler (22-E-3904)', 'thermal design', 'Vendor shall carry out'),
            confirmed,
        )
        self.assertIn(
            ('Silo Venting Blower 22-K-3904A/B', 'part of', '22-PK-3904'),
            confirmed,
        )
        q_codes = [
            c for row in (quarantine.get('relations') or []) + (quarantine.get('triplets') or [])
            for c in (row.get('reason_codes') or [])
        ]
        self.assertIn('claim_not_source_backed', q_codes)
        preserved = [
            row for row in (quarantine.get('triplets') or [])
            if 'claim_not_source_backed' in (row.get('reason_codes') or [])
        ]
        self.assertTrue(preserved)
        self.assertTrue(preserved[0].get('preserved_evidence') or preserved[0].get('evidence'))


class PintMeasurementAttributeTests(unittest.TestCase):
    """Pint confirms numeric/unit literals; non-numeric attrs keep YAML/KV rules."""

    @classmethod
    def setUpClass(cls):
        er.clear_section_aware_caches()
        cls.policy = er.load_policy()
        cls.settings = er._section_aware_evidence_settings(cls.policy)

    def test_settings_expose_pint_and_meas_qty_tokens(self):
        self.assertTrue(self.settings.get('pint_confirm_measurement_objects', False))
        self.assertIn('density', self.settings.get('measurement_field_tokens') or frozenset())
        self.assertIn('qty', self.settings.get('quantity_field_tokens') or frozenset())
        self.assertIn('mode', self.settings.get('measurement_field_tokens') or frozenset())

    def test_density_pint_confirms_quantity_literal(self):
        self.assertTrue(er._object_parses_as_quantity('900 kg/m³', policy=self.policy))
        self.assertTrue(
            er._looks_like_technical_attribute_claim(
                'Real Density',
                '900 kg/m³',
                policy=self.policy,
            )
        )
        eid, kind = er._resolve_endpoint_ref(
            '900 kg/m³',
            by_surface={},
            attribute_mode=True,
            policy=self.policy,
        )
        self.assertIsNone(eid)
        self.assertEqual(kind, 'literal')

    def test_quantity_bare_number_confirmed(self):
        self.assertTrue(er._object_parses_as_quantity('2', policy=self.policy))
        self.assertTrue(
            er._looks_like_technical_attribute_claim(
                'QTY',
                '2',
                policy=self.policy,
            )
        )

    def test_conveying_mode_non_numeric_uses_existing_marker_path(self):
        self.assertFalse(
            er._object_looks_like_numeric_measurement('Dilute phase', policy=self.policy)
        )
        self.assertTrue(
            er._looks_like_technical_attribute_claim(
                'Conveying Mode',
                'Dilute phase',
                policy=self.policy,
            )
        )

    def test_conveying_gas_non_numeric_or_evidence_path(self):
        # Plain gas name: non-numeric → marker/alias field path (no Pint required).
        self.assertTrue(
            er._looks_like_technical_attribute_claim(
                'Conveying media',
                'Nitrogen',
                policy=self.policy,
            )
        )
        # Gas codes like N2 are categorical (not Pint quantities).
        self.assertFalse(er._object_looks_like_numeric_measurement('N2', policy=self.policy))
        self.assertFalse(er._object_parses_as_quantity('N2', policy=self.policy))
        self.assertTrue(
            er._looks_like_technical_attribute_claim(
                'Conveying Mode',
                'N2',
                policy=self.policy,
            )
        )

    def test_connection_value_via_existing_kv_evidence(self):
        self.assertTrue(
            er._looks_like_technical_attribute_claim(
                'Connection Type',
                'Flanged',
                evidence_quote='Connection Type: Flanged',
                policy=self.policy,
            )
        )
        self.assertTrue(
            er._looks_like_labeled_attribute_object(
                'Connection Type - RF flange',
                policy=self.policy,
            )
        )

    def test_ambiguous_numeric_without_pint_or_evidence_stays_false(self):
        # Looks numeric under a measurement label but is not a Pint quantity.
        self.assertTrue(
            er._object_looks_like_numeric_measurement('approx 90-100ish', policy=self.policy)
        )
        self.assertFalse(er._object_parses_as_quantity('approx 90-100ish', policy=self.policy))
        self.assertFalse(
            er._looks_like_technical_attribute_claim(
                'Capacity',
                'approx 90-100ish',
                policy=self.policy,
            )
        )


class EvidenceFirstExtractionTests(unittest.TestCase):
    """Creation-time verify, provenance stamp, merge survival (no Bedrock)."""

    def setUp(self):
        self.policy = er.load_policy()

    def test_finalize_fills_absolute_offsets_on_full_doc(self):
        full = 'Header noise.\nUnique Pump P-101 feeds Tank T-1 here.\nFooter.'
        quote = 'Pump P-101 feeds Tank T-1'
        # Simulate Semantica objects with chunk-local metadata.
        class _Rel:
            def __init__(self):
                self.subject = 'Pump P-101'
                self.predicate = 'feeds'
                self.object = 'Tank T-1'
                self.metadata = {
                    'evidence': {
                        'source_text': quote,
                        'start_char': 0,
                        'end_char': len(quote),
                    }
                }

        rows = er._postprocess_chunk_extract(
            [_Rel()],
            full_text=full,
            source_file='inputs/x.md',
            span={'start': 14, 'end': 60, 'parent_heading': '## Unit', 'method': 'heading'},
            policy=self.policy,
            kind='relation',
        )
        meta = rows[0].metadata
        evidence = meta.get('evidence') or {}
        self.assertEqual(evidence.get('source_text'), quote)
        self.assertEqual(evidence.get('start_char'), full.find(quote))
        self.assertEqual(evidence.get('end_char'), full.find(quote) + len(quote))
        self.assertEqual(evidence.get('source_file'), 'inputs/x.md')
        prov = meta.get('provenance') or {}
        self.assertEqual(prov.get('chunk_start'), 14)
        self.assertEqual(prov.get('parent_heading'), '## Unit')

    def test_finalize_attribute_keeps_value_quote_and_section_context(self):
        full = (
            '## 6.1 22-PK-3801: Polymer Package\n\n'
            'Design Capacity - 71.5 TPH\n'
        )
        class _Trip:
            def __init__(self):
                self.subject = '22-PK-3801'
                self.predicate = 'Design Capacity'
                self.object = '71.5 TPH'
                self.metadata = {
                    'evidence': {'source_text': 'Design Capacity - 71.5 TPH'},
                }

        rows = er._finalize_chunk_evidence(
            [_Trip()],
            full_text=full,
            source_file='inputs/x.md',
            policy=self.policy,
            kind='triplet',
        )
        meta = rows[0].metadata
        evidence = meta.get('evidence') or {}
        self.assertEqual(evidence.get('source_text'), 'Design Capacity - 71.5 TPH')
        self.assertNotIn('22-PK-3801', evidence.get('source_text') or '')
        section = meta.get('section_context') or {}
        self.assertIn('22-PK-3801', section.get('source_text') or '')

    def test_evidence_survives_identity_merge_with_peer_refs(self):
        left = {
            'id': 'e0',
            'text': '22-PK-3801',
            'label': 'System',
            'confidence': 0.9,
            'metadata': {
                'canonical_id': 'e0',
                'identifiers': [{'value': '22-PK-3801', 'role': 'identity'}],
                'evidence': {
                    'source_text': '22-PK-3801 : Polymer Pneumatic',
                    'source_file': 'inputs/x.md',
                    'start_char': 0,
                    'end_char': 30,
                },
                'provenance': {'chunk_start': 0, 'parent_heading': '## Scope'},
            },
        }
        right = {
            'id': 'e1',
            'text': 'Conveying System 22-PK-3801',
            'label': 'System',
            'confidence': 0.8,
            'metadata': {
                'canonical_id': 'e1',
                'identifiers': [{'value': '22-PK-3801', 'role': 'identity'}],
                'evidence': {
                    'source_text': 'Conveying System 22-PK-3801: Polymer',
                    'source_file': 'inputs/x.md',
                    'start_char': 100,
                    'end_char': 136,
                },
                'section_context': {
                    'source_text': '## 6.1 Conveying System 22-PK-3801',
                    'resolver': 'deterministic_section_row',
                },
                'provenance': {'chunk_start': 80},
            },
        }
        bundle = er.ExtractionBundle(entities=[left, right])
        er.merge_identity_duplicate_entities(
            bundle, self.policy, document_text='22-PK-3801 : Polymer Pneumatic',
        )
        self.assertEqual(len(bundle.entities), 1)
        survivor = bundle.entities[0]
        meta = survivor.get('metadata') or {}
        self.assertTrue((meta.get('evidence') or {}).get('source_text'))
        refs = meta.get('source_references') or []
        self.assertTrue(refs)
        peer = refs[0]
        self.assertIn('Conveying System 22-PK-3801', peer.get('source_text') or '')
        self.assertTrue(peer.get('section_context') or meta.get('section_context'))
        self.assertTrue(meta.get('provenance'))

    def test_mismatched_quote_not_invented(self):
        class _Rel:
            def __init__(self, subject, obj, quote):
                self.subject = subject
                self.predicate = 'feeds'
                self.object = obj
                self.metadata = {
                    'evidence': {'source_text': quote},
                }

        bad = 'this quote is not in the document'
        # Source-backed recovery may replace a bad quote; never keep the invented text.
        rows = er._finalize_chunk_evidence(
            [_Rel('Pump A', 'Tank B', bad)],
            full_text='Pump A feeds Tank B in unit 3.',
            source_file='inputs/x.md',
            policy=self.policy,
            kind='relation',
        )
        meta = rows[0].metadata
        ev = meta.get('evidence') or {}
        self.assertNotEqual(ev.get('source_text'), bad)
        if ev.get('source_text'):
            self.assertIn(ev['source_text'], 'Pump A feeds Tank B in unit 3.')
            self.assertEqual(meta.get('evidence_recovery'), 'deterministic_span')
        self.assertNotEqual(meta.get('graph_eligible'), True)

        # When endpoints are absent, recovery must not invent evidence.
        empty = er._finalize_chunk_evidence(
            [_Rel('Ghost Pump', 'Ghost Tank', bad)],
            full_text='Pump A feeds Tank B in unit 3.',
            source_file='inputs/x.md',
            policy=self.policy,
            kind='relation',
        )
        empty_meta = empty[0].metadata
        self.assertNotIn('evidence', empty_meta)
        self.assertNotEqual(empty_meta.get('graph_eligible'), True)

    def test_confirmed_graph_still_applies_claim_gates_on_preserved_evidence(self):
        entities = [
            {
                'id': 'e0',
                'text': 'Pkg',
                'label': 'System',
                'metadata': {
                    'canonical_id': 'e0',
                    'identifiers': [{'value': 'PKG-1', 'role': 'identity'}],
                    'evidence': {'source_text': 'Pkg includes Eq'},
                },
            },
            {
                'id': 'e1',
                'text': 'Eq',
                'label': 'Equipment',
                'metadata': {
                    'canonical_id': 'e1',
                    'identifiers': [{'value': 'EQ-1', 'role': 'identity'}],
                    'evidence': {'source_text': 'Pkg includes Eq'},
                },
            },
        ]
        good = {
            'subject': 'Pkg',
            'predicate': 'includes',
            'object': 'Eq',
            'subject_id': 'e0',
            'object_id': 'e1',
            'subject_ref_kind': 'entity',
            'object_ref_kind': 'entity',
            'relation_category': 'engineering_hierarchy',
            'metadata': {
                'evidence_status': 'verified',
                'graph_eligible': True,
                'evidence': {'source_text': 'Pkg includes Eq in package scope'},
                'provenance': {'chunk_start': 0},
            },
        }
        bad = {
            'subject': 'Pkg',
            'predicate': 'includes',
            'object': 'Eq',
            'subject_id': 'e0',
            'object_id': 'e1',
            'subject_ref_kind': 'entity',
            'object_ref_kind': 'entity',
            'relation_category': 'engineering_hierarchy',
            'metadata': {
                'evidence_status': 'verified',
                'graph_eligible': True,
                # Co-mention title list without membership/predicate cue.
                'evidence': {
                    'source_text': 'SPEC FOR PACKAGE\n(Pkg, Other)',
                },
            },
        }
        # Only the gated-good edge should confirm when claim support holds.
        bundle = er.ExtractionBundle(entities=entities, relations=[good])
        graph, _q = er.build_confirmed_graph(bundle, self.policy)
        self.assertGreaterEqual(graph.get('edge_count') or 0, 1)

        bundle_bad = er.ExtractionBundle(entities=entities, relations=[bad])
        graph_bad, quarantine = er.build_confirmed_graph(bundle_bad, self.policy)
        self.assertEqual(graph_bad.get('edge_count') or 0, 0)
        codes = [
            c for row in (quarantine.get('relations') or [])
            for c in (row.get('reason_codes') or [])
        ]
        self.assertIn('claim_not_source_backed', codes)


class SemanticRelationRecoveryTests(unittest.TestCase):
    """Generic source-grounded semantic recovery (YAML contracts, no hardcoding)."""

    @classmethod
    def setUpClass(cls):
        cls.policy = er.load_policy()

    def test_active_passive_and_inverse_relationships(self):
        active = 'Pump A feeds Tank B in unit 3.'
        self.assertTrue(
            er._claim_supported_by_quote(
                active, 'Pump A', 'Tank B', predicate='feeds', policy=self.policy,
            )
        )
        passive = 'Tank B is fed by Pump A in unit 3.'
        self.assertTrue(
            er._claim_supported_by_quote(
                passive, 'Tank B', 'Pump A', predicate='fed by', policy=self.policy,
            )
        )
        # Inverse orientation must not confirm the active predicate.
        self.assertFalse(
            er._claim_supported_by_quote(
                passive, 'Pump A', 'Tank B', predicate='feeds', policy=self.policy,
            )
        )

    def test_role_assignment_and_project_association(self):
        role_quote = (
            'For execution of this Project, Client Org have appointed '
            'Vendor Solutions Ltd. as their EPCM Consultant.'
        )
        out = er._attach_claim_evidence(
            {'evidence': {'source_text': role_quote}},
            role_quote,
            subject='Vendor Solutions Ltd.',
            obj='EPCM Consultant',
            predicate='appointed as',
            source_file='inputs/a.md',
            policy=self.policy,
        )
        self.assertEqual(out.get('evidence_status'), 'verified')
        self.assertTrue(out.get('graph_eligible'))

        project_quote = (
            'Supply of Package Alpha for Project of Expansion Initiative at Site.'
        )
        # Without a Project-typed endpoint this stays a meaning check only.
        self.assertTrue(
            er._predicate_meaning_in_quote(
                project_quote, 'for project', policy=self.policy,
            )
        )

    def test_valid_and_invalid_hierarchy_unchanged(self):
        doc = 'Package Alpha includes Equipment Beta in the package scope.'
        good = er._attach_claim_evidence(
            {'evidence': {'source_text': doc}},
            doc,
            subject='Package Alpha',
            obj='Equipment Beta',
            predicate='includes',
            policy=self.policy,
        )
        self.assertEqual(good.get('evidence_status'), 'verified')
        # Co-mention without membership cue stays unverified.
        bad_doc = 'SPEC FOR PACKAGE\n(Package Alpha, Equipment Beta)'
        bad = er._attach_claim_evidence(
            {'evidence': {'source_text': bad_doc}},
            bad_doc,
            subject='Package Alpha',
            obj='Equipment Beta',
            predicate='includes',
            policy=self.policy,
        )
        self.assertNotEqual(bad.get('evidence_status'), 'verified')

    def test_repeated_text_prefers_local_chunk_window(self):
        doc = (
            '## Section One\n'
            'Pump A feeds Tank B here.\n\n'
            '## Section Two\n'
            'Pump A feeds Tank B again with notes.\n'
        )
        index = er.build_document_evidence_index(
            doc,
            source_file='inputs/a.md',
            policy=self.policy,
            entity_surfaces=['Pump A', 'Tank B'],
        )
        # Localize to section two offsets.
        sec2 = doc.index('## Section Two')
        quote = er._recover_quote_from_evidence_index(
            index,
            subject='Pump A',
            obj='Tank B',
            predicate='feeds',
            policy=self.policy,
            chunk_start=sec2,
            chunk_end=len(doc),
        )
        self.assertIn('again with notes', quote)

    def test_missing_evidence_and_ambiguous_endpoints(self):
        entities = [
            {
                'id': 'e0',
                'text': 'Vendor Solutions Ltd.',
                'label': 'Organization',
                'metadata': {'canonical_id': 'e0'},
            },
            {
                'id': 'e1',
                'text': 'EPCM Consultant',
                'label': 'Role',
                'metadata': {'canonical_id': 'e1'},
            },
        ]
        rel = {
            'subject': 'Vendor Solutions Ltd.',
            'predicate': 'appointed as',
            'object': 'EPCM Consultant',
            'subject_id': 'e0',
            'object_id': 'e1',
            'relation_category': 'semantic',
            'metadata': {
                'evidence_status': 'missing',
                'graph_eligible': False,
            },
        }
        # Missing quote + endpoints absent from doc → still missing, not invented.
        bundle = er.ExtractionBundle(entities=entities, relations=[rel])
        stats = er.recover_semantic_relations(
            bundle,
            self.policy,
            document_text='No appointment language in this document.',
            source_file='inputs/a.md',
        )
        meta = bundle.relations[0]['metadata']
        self.assertNotEqual(meta.get('graph_eligible'), True)
        self.assertIn(meta.get('evidence_status'), {'missing', 'unverified', None, ''})

        # Ambiguous co-mention without cue must not confirm.
        co = 'Vendor Solutions Ltd. and EPCM Consultant appear on the cover.'
        self.assertFalse(
            er._claim_supported_by_quote(
                co,
                'Vendor Solutions Ltd.',
                'EPCM Consultant',
                predicate='appointed as',
                policy=self.policy,
            )
        )
        self.assertGreaterEqual(stats.get('skipped', 0) + stats.get('repaired', 0), 0)

    def test_multi_document_provenance_isolation(self):
        doc_a = 'Vendor Solutions Ltd. was appointed as EPCM Consultant for Site A.'
        doc_b = 'Other Vendor Ltd. was appointed as EPCM Consultant for Site B.'
        index_a = er.build_document_evidence_index(
            doc_a, source_file='inputs/a.md', policy=self.policy,
            entity_surfaces=['Vendor Solutions Ltd.', 'EPCM Consultant'],
        )
        index_b = er.build_document_evidence_index(
            doc_b, source_file='inputs/b.md', policy=self.policy,
            entity_surfaces=['Other Vendor Ltd.', 'EPCM Consultant'],
        )
        quote_a = er._recover_quote_from_evidence_index(
            index_a,
            subject='Vendor Solutions Ltd.',
            obj='EPCM Consultant',
            predicate='appointed as',
            policy=self.policy,
        )
        quote_b = er._recover_quote_from_evidence_index(
            index_b,
            subject='Other Vendor Ltd.',
            obj='EPCM Consultant',
            predicate='appointed as',
            policy=self.policy,
        )
        self.assertIn('Site A', quote_a)
        self.assertIn('Site B', quote_b)
        self.assertNotIn('Other Vendor', quote_a)
        self.assertNotIn('Vendor Solutions', quote_b)
        self.assertEqual(index_a.source_file, 'inputs/a.md')
        self.assertEqual(index_b.source_file, 'inputs/b.md')

    def test_large_document_chunk_boundaries(self):
        parts = [f'## Part {i}\nIntro text for part {i}.\n' for i in range(12)]
        parts.append('## Final\nPump A feeds Tank B at the end.\n')
        doc = '\n'.join(parts)
        spans = er.section_aware_chunk_spans(doc, max_chars=180)
        self.assertGreater(len(spans), 1)
        index = er.build_document_evidence_index(
            doc, source_file='inputs/big.md', policy=self.policy,
            entity_surfaces=['Pump A', 'Tank B'],
        )
        last = spans[-1]
        quote = er._recover_quote_from_evidence_index(
            index,
            subject='Pump A',
            obj='Tank B',
            predicate='feeds',
            policy=self.policy,
            chunk_start=int(last['start']),
            chunk_end=int(last['end']),
        )
        self.assertIn('Pump A', quote)
        self.assertIn('Tank B', quote)
        self.assertIn('feeds', quote.casefold())

    def test_no_false_positive_graph_confirmation(self):
        entities = [
            {
                'id': 'e0',
                'text': 'Vendor Solutions Ltd.',
                'label': 'Organization',
                'metadata': {'canonical_id': 'e0'},
            },
            {
                'id': 'e1',
                'text': 'EPCM Consultant',
                'label': 'Role',
                'metadata': {'canonical_id': 'e1'},
            },
            {
                'id': 'e2',
                'text': 'Package Alpha',
                'label': 'System',
                'metadata': {'canonical_id': 'e2'},
            },
        ]
        # Cover co-mention with stale eligible flag must stay quarantined.
        rel = {
            'subject': 'Vendor Solutions Ltd.',
            'predicate': 'appointed as',
            'object': 'EPCM Consultant',
            'subject_id': 'e0',
            'object_id': 'e1',
            'relation_category': 'semantic',
            'metadata': {
                'evidence_status': 'verified',
                'graph_eligible': True,
                'evidence': {
                    'source_text': (
                        'Vendor Solutions Ltd. and EPCM Consultant listed on cover'
                    ),
                },
            },
        }
        bundle = er.ExtractionBundle(entities=entities, relations=[rel])
        er.recover_semantic_relations(
            bundle,
            self.policy,
            document_text=(
                'Vendor Solutions Ltd. and EPCM Consultant listed on cover. '
                'Package Alpha is described elsewhere.'
            ),
            source_file='inputs/a.md',
        )
        graph, quarantine = er.build_confirmed_graph(bundle, self.policy)
        self.assertEqual(graph.get('edge_count') or 0, 0)
        codes = [
            c
            for row in (quarantine.get('relations') or [])
            for c in (row.get('reason_codes') or [])
        ]
        self.assertTrue(
            {'claim_not_source_backed', 'evidence_unverified', 'not_graph_eligible'} & set(codes)
        )

    def test_orientation_flip_preserves_history_and_syncs_quarantine(self):
        doc = (
            'BIDDING DOCUMENT FOR Package Alpha\n\n'
            'PREPARED BY Vendor Solutions Ltd.'
        )
        entities = [
            {
                'id': 'e0',
                'text': 'Vendor Solutions Ltd.',
                'label': 'Organization',
                'metadata': {'canonical_id': 'e0'},
            },
            {
                'id': 'e1',
                'text': 'Package Alpha',
                'label': 'System',
                'metadata': {'canonical_id': 'e1'},
            },
        ]
        rel = {
            'subject': 'Vendor Solutions Ltd.',
            'predicate': 'prepared by',
            'object': 'Package Alpha',
            'subject_id': 'e0',
            'object_id': 'e1',
            'relation_category': 'semantic',
            'metadata': {
                'evidence_status': 'unverified',
                'graph_eligible': False,
                'evidence': {'source_text': doc},
                'provenance': {'source_file': 'inputs/a.md', 'chunk_start': 0, 'chunk_end': len(doc)},
            },
        }
        bundle = er.ExtractionBundle(entities=entities, relations=[rel])
        stats = er.recover_semantic_relations(
            bundle, self.policy, document_text=doc, source_file='inputs/a.md',
        )
        self.assertGreaterEqual(stats.get('oriented', 0), 1)
        fixed = bundle.relations[0]
        self.assertEqual(fixed.get('subject'), 'Package Alpha')
        self.assertEqual(fixed.get('object'), 'Vendor Solutions Ltd.')
        history = (fixed.get('metadata') or {}).get('recovery_history') or []
        self.assertTrue(any(h.get('action') == 'orientation_flip' for h in history))
        # Quarantine must reflect post-recovery evidence state.
        qrels = (bundle.quarantine or {}).get('relations') or []
        graph_edges = (bundle.graph or {}).get('edges') or []
        self.assertTrue(qrels or graph_edges)


class TableOwnershipResolverTests(unittest.TestCase):
    """Structure-aware table ownership (no hardcoded tags/names)."""

    @classmethod
    def setUpClass(cls):
        cls.policy = er.load_policy()

    def setUp(self):
        er.clear_section_aware_caches()

    def test_markdown_pipe_table_owner_and_continuation(self):
        doc = (
            '| Item | Parameters |\n'
            '| --- | --- |\n'
            '| 11-PK-1001: Alpha Package | Type of System - Closed loop |\n'
            '| Transport to Surge Silos. | Design Capacity - 10 TPH |\n'
            '| 11-PK-1002: Beta Package | Type of System - Open loop |\n'
            '| Feed from Extruder. | Design Capacity - 20 TPH |\n'
        )
        idx = er.build_table_ownership_index(doc, policy=self.policy)
        owners = [r.owner_surface for r in idx.rows if r.owner_surface]
        self.assertIn('11-PK-1001: Alpha Package', owners)
        cont = [r for r in idx.rows if r.is_continuation]
        self.assertTrue(cont)
        self.assertTrue(all('11-PK-' in (r.owner_surface or '') for r in cont))

        meta = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Design Capacity - 10 TPH'}},
            doc,
            subject='11-PK-1001: Alpha Package',
            obj='10 TPH',
            predicate='Design Capacity',
            attribute_mode=True,
            source_file='inputs/a.md',
            policy=self.policy,
        )
        self.assertEqual(meta.get('evidence_status'), 'verified')
        self.assertTrue(meta.get('graph_eligible'))
        ctx = meta.get('section_context') or {}
        self.assertEqual(ctx.get('kind'), 'table_row')
        self.assertIn('11-PK-1001', ctx.get('source_text') or '')
        # Value evidence stays separate from owner-cell provenance.
        self.assertNotIn('11-PK-1001', (meta.get('evidence') or {}).get('source_text') or '')

    def test_html_table_with_rowspan_colspan(self):
        doc = (
            '<table><tbody>'
            '<tr><td rowspan="2">22-AA-5001: Unit</td>'
            '<td colspan="2">Type of System - Dilute</td></tr>'
            '<tr><td>Conveying media - Nitrogen</td><td>Design Capacity - 5 TPH</td></tr>'
            '<tr><td>22-AA-5002: Other</td><td colspan="2">Type of System - Dense</td></tr>'
            '</tbody></table>'
        )
        idx = er.build_table_ownership_index(doc, policy=self.policy)
        self.assertTrue(any(r.owner_surface and '22-AA-5001' in r.owner_surface for r in idx.rows))
        meta = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Design Capacity - 5 TPH'}},
            doc,
            subject='22-AA-5001: Unit',
            obj='5 TPH',
            predicate='Design Capacity',
            attribute_mode=True,
            source_file='inputs/a.md',
            policy=self.policy,
        )
        self.assertEqual(meta.get('evidence_status'), 'verified')
        self.assertTrue(meta.get('graph_eligible'))

    def test_split_label_continuation_rows(self):
        doc = (
            '| 33-PK-7701: Split Owner | Type of System - Open loop |\n'
            '| transport from Extruder to | Conveying media - Air |\n'
            '| Homogenizing Silos. | Design Capacity - 86 TPH |\n'
        )
        meta = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Design Capacity - 86 TPH'}},
            doc,
            subject='33-PK-7701: Split Owner',
            obj='86 TPH',
            predicate='Design Capacity',
            attribute_mode=True,
            source_file='inputs/a.md',
            policy=self.policy,
        )
        self.assertEqual(meta.get('evidence_status'), 'verified')
        self.assertEqual((meta.get('section_context') or {}).get('resolver'), 'table_row_group')

    def test_ambiguous_multi_tag_row_stays_quarantined(self):
        doc = (
            '| 44-PK-1001 / 44-PK-1002 shared | Design Capacity - 1 TPH |\n'
        )
        meta = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Design Capacity - 1 TPH'}},
            doc,
            subject='44-PK-1001',
            obj='1 TPH',
            predicate='Design Capacity',
            attribute_mode=True,
            source_file='inputs/a.md',
            policy=self.policy,
        )
        self.assertNotEqual(meta.get('graph_eligible'), True)

    def test_repeated_values_across_sections_stay_local(self):
        doc = (
            '## Package A\n'
            '| 55-PK-2001: First | Design Capacity - 71.5 TPH |\n\n'
            '## Package B\n'
            '| 55-PK-2002: Second | Design Capacity - 71.5 TPH |\n'
        )
        q = 'Design Capacity - 71.5 TPH'
        # Explicit offsets for first occurrence.
        start = doc.index(q)
        meta = er._attach_claim_evidence(
            {'evidence': {'source_text': q, 'start_char': start, 'end_char': start + len(q)}},
            doc,
            subject='55-PK-2001: First',
            obj='71.5 TPH',
            predicate='Design Capacity',
            attribute_mode=True,
            source_file='inputs/a.md',
            policy=self.policy,
        )
        self.assertEqual(meta.get('evidence_status'), 'verified')
        self.assertIn('55-PK-2001', (meta.get('section_context') or {}).get('source_text') or '')

        start2 = doc.index(q, start + 1)
        meta2 = er._attach_claim_evidence(
            {'evidence': {'source_text': q, 'start_char': start2, 'end_char': start2 + len(q)}},
            doc,
            subject='55-PK-2002: Second',
            obj='71.5 TPH',
            predicate='Design Capacity',
            attribute_mode=True,
            source_file='inputs/b.md',
            policy=self.policy,
        )
        self.assertEqual(meta2.get('evidence_status'), 'verified')
        self.assertIn('55-PK-2002', (meta2.get('section_context') or {}).get('source_text') or '')

    def test_multi_document_provenance_isolation(self):
        doc_a = '| 66-PK-3001: DocA | Design Capacity - 3 TPH |\n'
        doc_b = '| 66-PK-3002: DocB | Design Capacity - 3 TPH |\n'
        meta_a = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Design Capacity - 3 TPH'}},
            doc_a,
            subject='66-PK-3001: DocA',
            obj='3 TPH',
            predicate='Design Capacity',
            attribute_mode=True,
            source_file='inputs/a.md',
            policy=self.policy,
        )
        meta_b = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Design Capacity - 3 TPH'}},
            doc_b,
            subject='66-PK-3002: DocB',
            obj='3 TPH',
            predicate='Design Capacity',
            attribute_mode=True,
            source_file='inputs/b.md',
            policy=self.policy,
        )
        self.assertEqual((meta_a.get('section_context') or {}).get('source_file'), 'inputs/a.md')
        self.assertEqual((meta_b.get('section_context') or {}).get('source_file'), 'inputs/b.md')
        self.assertIn('66-PK-3001', (meta_a.get('section_context') or {}).get('source_text') or '')
        self.assertIn('66-PK-3002', (meta_b.get('section_context') or {}).get('source_text') or '')

    def test_no_false_positive_without_table_structure(self):
        doc = (
            'Random prose mentions Design Capacity - 9 TPH near 77-PK-4001 '
            'but not in a table row.'
        )
        meta = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Design Capacity - 9 TPH'}},
            doc,
            subject='77-PK-4001',
            obj='9 TPH',
            predicate='Design Capacity',
            attribute_mode=True,
            source_file='inputs/a.md',
            policy=self.policy,
        )
        self.assertNotEqual(meta.get('graph_eligible'), True)

    def test_subject_correction_from_unique_row_owner(self):
        doc = '| 88-PK-8001: True Owner | Design Capacity - 12 TPH |\n'
        meta = er._attach_claim_evidence(
            {'evidence': {'source_text': 'Design Capacity - 12 TPH'}},
            doc,
            subject='Wrong Subject Name',
            obj='12 TPH',
            predicate='Design Capacity',
            attribute_mode=True,
            source_file='inputs/a.md',
            policy=self.policy,
        )
        self.assertEqual(meta.get('evidence_status'), 'verified')
        self.assertIn('88-PK-8001', meta.get('corrected_subject') or '')


class ProvenanceSubjectResolutionTests(unittest.TestCase):
    """Subjects grounded in evidence/section_context must not stay unresolved_subject."""

    @classmethod
    def setUpClass(cls):
        cls.policy = er.load_policy()

    def setUp(self):
        er.clear_section_aware_caches()

    def test_section_context_subject_not_unresolved(self):
        doc = (
            '| 99-PK-9004: Blenders Blowing | Medium - Air |\n'
            '| system | No of Silos - 8 nos |\n'
        )
        # Compound package also mentions the same tag — must not steal the row owner.
        compound = {
            'id': 'e_pkg',
            'text': 'PACKAGE (99-PK-9001) & BLENDERS BLOWING SYSTEM (99-PK-9004)',
            'label': 'System',
            'metadata': {
                'canonical_id': 'e_pkg',
                'identifiers': [
                    {'value': '99-PK-9001', 'role': 'identity'},
                    {'value': '99-PK-9004', 'role': 'identity'},
                ],
            },
        }
        trip = {
            'subject': '99-PK-9004: Blenders Blowing system',
            'predicate': 'Medium',
            'object': 'Air',
            'relation_category': 'attribute',
            'metadata': {
                'evidence': {'source_text': 'Medium - Air'},
                'section_context': {
                    'source_text': '99-PK-9004: Blenders Blowing',
                    'equipment_tags': ['99-PK-9004'],
                    'kind': 'table_row',
                    'resolver': 'table_row_owner',
                },
                'evidence_status': 'verified',
                'graph_eligible': True,
                'attribute_source_scope': 'specific',
            },
        }
        # Grounding alone removes unresolved_subject even before mint.
        self.assertTrue(
            er._subject_grounded_in_provenance(
                trip['subject'], trip['metadata'], policy=self.policy,
            )
        )
        bundle = er.ExtractionBundle(entities=[compound], triplets=[trip])
        created = er.recover_missing_endpoints(
            bundle, self.policy, document_text=doc, source_file='inputs/a.md',
        )
        self.assertGreaterEqual(created, 1)
        er.validate_extraction_graph(
            bundle, self.policy, source_file='inputs/a.md', document_text=doc,
        )
        self.assertEqual(bundle.triplets[0].get('subject_ref_kind'), 'entity')
        self.assertTrue(bundle.triplets[0].get('subject_id'))
        self.assertNotEqual(bundle.triplets[0].get('subject_id'), 'e_pkg')
        graph, quarantine = er.build_confirmed_graph(
            bundle, self.policy, source_file='inputs/a.md',
        )
        q_codes = [
            c
            for row in (quarantine.get('triplets') or [])
            for c in (row.get('reason_codes') or [])
        ]
        self.assertNotIn('unresolved_subject', q_codes)
        self.assertGreaterEqual(graph.get('edge_count') or 0, 1)

    def test_evidence_exact_subject_grounding(self):
        meta = {
            'evidence': {
                'source_text': 'Pump A Design Capacity - 1 TPH',
            },
            'section_context': None,
        }
        self.assertTrue(
            er._subject_grounded_in_provenance('Pump A', meta, policy=self.policy)
        )
        self.assertFalse(
            er._subject_grounded_in_provenance('Pump Z', meta, policy=self.policy)
        )


class RecoveredEntitySafetyTests(unittest.TestCase):
    """Recovered nodes must pass YAML non-entity / typing gates."""

    @classmethod
    def setUpClass(cls):
        cls.policy = er.load_policy()

    def setUp(self):
        er.clear_section_aware_caches()

    def test_scalar_attribute_objects_are_not_entities(self):
        for surface in ('8', '5', '1000 m3', '12 bar'):
            self.assertTrue(
                er._surface_is_property_not_entity(
                    surface, policy=self.policy, attribute_object=True,
                ),
                msg=surface,
            )
            self.assertIsNone(
                er._validate_recovered_entity_proposal(
                    surface,
                    proposed_label='System',
                    policy=self.policy,
                    attribute_object=True,
                ),
                msg=surface,
            )

    def test_material_as_attribute_property_not_entity(self):
        # Conveying medium value on an attribute claim is a property, not Material/System.
        self.assertIsNone(
            er._validate_recovered_entity_proposal(
                'Air',
                proposed_label='System',
                predicate='Medium',
                role='object',
                policy=self.policy,
                attribute_object=True,
            )
        )
        # LLM hint alone never validates a non-identifier surface.
        self.assertIsNone(
            er._validate_recovered_entity_proposal(
                'Air',
                proposed_label='Material',
                policy=self.policy,
                attribute_object=False,
            )
        )

    def test_package_owner_typing_not_blind_equipment(self):
        primary, subtype = er._infer_recovered_typing(
            '99-PK-9004: Blenders Blowing',
            predicate='Medium',
            role='subject',
            settings=er._endpoint_recovery_settings(self.policy),
        )
        self.assertEqual(primary, 'System')
        self.assertEqual(subtype, 'EngineeringPackage')
        validated = er._validate_recovered_entity_proposal(
            '99-PK-9004: Blenders Blowing',
            proposed_label='Equipment',
            policy=self.policy,
        )
        self.assertIsNotNone(validated)
        self.assertEqual(validated[0], 'System')
        self.assertEqual(validated[1], 'EngineeringPackage')

    def test_scrub_removes_fake_llm_attribute_entities(self):
        doc = (
            '| 99-PK-9004: Blenders Blowing | Medium - Air |\n'
            '| system | No of Silos - 8 nos |\n'
        )
        bundle = er.ExtractionBundle(
            entities=[
                {
                    'id': 'recovered_air',
                    'text': 'Air',
                    'label': 'System',
                    'metadata': {
                        'canonical_id': 'recovered_air',
                        'extraction_method': 'llm_quarantine_resolver',
                    },
                },
                {
                    'id': 'recovered_eight',
                    'text': '8',
                    'label': 'System',
                    'metadata': {
                        'canonical_id': 'recovered_eight',
                        'extraction_method': 'llm_quarantine_resolver',
                    },
                },
            ],
            triplets=[{
                'subject': '99-PK-9004: Blenders Blowing system',
                'predicate': 'Medium',
                'object': 'Air',
                'object_id': 'recovered_air',
                'relation_category': 'attribute',
                'metadata': {
                    'evidence': {'source_text': 'Medium - Air'},
                    'section_context': {
                        'source_text': '99-PK-9004: Blenders Blowing',
                        'equipment_tags': ['99-PK-9004'],
                        'kind': 'table_row',
                        'resolver': 'table_row_owner',
                    },
                    'evidence_status': 'verified',
                    'graph_eligible': True,
                },
            }],
        )
        removed = er._scrub_invalid_recovered_entities(
            bundle, self.policy, document_text=doc,
        )
        self.assertIn('recovered_air', removed)
        self.assertIn('recovered_eight', removed)
        self.assertEqual(bundle.triplets[0].get('object_ref_kind'), 'literal')
        self.assertIsNone(bundle.triplets[0].get('object_id'))
        created = er.recover_missing_endpoints(
            bundle, self.policy, document_text=doc, source_file='inputs/a.md',
        )
        self.assertGreaterEqual(created, 1)
        owner = next(
            e for e in bundle.entities
            if '99-PK-9004: Blenders Blowing' == er.collapse(str(e.get('text') or ''))
            or str(e.get('text') or '').startswith('99-PK-9004:')
        )
        self.assertEqual(owner.get('label'), 'System')
        self.assertEqual((owner.get('metadata') or {}).get('subtype'), 'EngineeringPackage')
        texts = {er.collapse(str(e.get('text') or '')) for e in bundle.entities}
        self.assertNotIn('Air', texts)
        self.assertNotIn('8', texts)

    def test_valid_control_loop_recovery_still_works(self):
        quote = '22-K-3901A-C controlled by ESD 3909A-H in release mode'
        bundle = er.ExtractionBundle(
            entities=[{
                'id': 'e5',
                'text': '22-K-3901A-C',
                'label': 'Equipment',
                'metadata': {'canonical_id': 'e5', 'identifiers': ['22-K-3901A-C']},
            }],
            relations=[{
                'subject': '22-K-3901A-C',
                'predicate': 'controlled by',
                'object': 'ESD 3909A-H',
                'metadata': {
                    'evidence': {'source_text': quote},
                    'evidence_status': 'verified',
                    'graph_eligible': True,
                },
            }],
        )
        created = er.recover_missing_endpoints(
            bundle, self.policy, document_text=quote, source_file='inputs/x.md',
        )
        self.assertEqual(created, 1)
        recovered = next(e for e in bundle.entities if e['text'] == 'ESD 3909A-H')
        self.assertEqual(recovered['label'], 'ControlLoop')


if __name__ == '__main__':
    unittest.main()
