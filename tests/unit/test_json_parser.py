"""Unit tests for JSON parser utilities."""

import json
import pytest

from pose_director_root.utils.json_parser import (
    parse_llm_json,
    extract_keypoints_from_dwpose,
)


class TestParseLlmJson:
    """Tests for parse_llm_json() with various LLM response formats."""

    def test_parse_direct_json_object(self):
        result = parse_llm_json('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_parse_direct_json_array(self):
        result = parse_llm_json('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_parse_markdown_json_code_block(self):
        response = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        result = parse_llm_json(response)
        assert result == {"key": "value"}

    def test_parse_markdown_code_block_uppercase(self):
        response = '```JSON\n{"key": "value"}\n```'
        result = parse_llm_json(response)
        assert result == {"key": "value"}

    def test_parse_markdown_code_block_no_language(self):
        response = '```\n{"key": "value"}\n```'
        result = parse_llm_json(response)
        assert result == {"key": "value"}

    def test_parse_trailing_commas(self):
        result = parse_llm_json('{"key": "value", "num": 42,}')
        assert result == {"key": "value", "num": 42}

    def test_parse_trailing_commas_in_array(self):
        result = parse_llm_json('[1, 2, 3,]')
        assert result == [1, 2, 3]

    def test_parse_embedded_json(self):
        response = 'I think the result is {"key": "value"} and that is all.'
        result = parse_llm_json(response)
        assert result == {"key": "value"}

    def test_parse_nested_json(self):
        data = {"outer": {"inner": [1, 2, 3]}}
        result = parse_llm_json(json.dumps(data))
        assert result == data

    def test_parse_keypoints_array(self):
        kpts = [100.0, 200.0, 0.9, 150.0, 250.0, 0.8]
        result = parse_llm_json(json.dumps(kpts))
        assert result == kpts

    def test_parse_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="Could not extract JSON"):
            parse_llm_json("this is not json at all no braces")

    def test_parse_empty_string_raises(self):
        with pytest.raises(ValueError):
            parse_llm_json("")

    def test_parse_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            parse_llm_json("   \n\n  ")

    def test_parse_js_style_comments(self):
        response = '{\n"key": "value", // a comment\n"num": 42\n}'
        result = parse_llm_json(response)
        assert result == {"key": "value", "num": 42}


class TestExtractKeypoints:
    """Tests for extract_keypoints_from_dwpose()."""

    def test_flat_list_passthrough(self):
        kpts = [100.0, 200.0, 0.9, 150.0, 250.0, 0.8] * 9  # 54 floats = 18 keypoints
        result, h, w = extract_keypoints_from_dwpose(kpts)
        assert len(result) == 54
        assert h is None
        assert w is None

    def test_dwpose_format_denormalized(self):
        """Test with normalized (0-1) coords — should convert to pixels."""
        kpts_18 = []
        for i in range(18):
            kpts_18.extend([0.5, 0.3, 0.9])  # All at center-ish
        pose_data = {
            "people": [{"pose_keypoints_2d": kpts_18}],
            "canvas_height": 1024,
            "canvas_width": 768,
        }
        result, h, w = extract_keypoints_from_dwpose(pose_data)
        assert h == 1024
        assert w == 768
        assert len(result) == 54
        # Coords should be denormalized
        assert result[0] == pytest.approx(0.5 * 768, abs=1)  # x
        assert result[1] == pytest.approx(0.3 * 1024, abs=1)  # y

    def test_dwpose_format_pixel_coords(self):
        """Test with already-pixel coords — should pass through unchanged."""
        kpts_18 = []
        for i in range(18):
            kpts_18.extend([500.0, 300.0, 0.9])  # Pixel coords > 1.0
        pose_data = {
            "people": [{"pose_keypoints_2d": kpts_18}],
            "canvas_height": 1024,
            "canvas_width": 768,
        }
        result, h, w = extract_keypoints_from_dwpose(pose_data)
        assert result[0] == 500.0  # Unchanged
        assert result[1] == 300.0

    def test_coco17_to_coco18_conversion(self):
        """Test COCO-17 (51 floats) → COCO-18 (54 floats) with Neck insertion."""
        kpts_17 = []
        for i in range(17):
            kpts_17.extend([float(i * 10), float(i * 10 + 5), 0.9])
        result, _, _ = extract_keypoints_from_dwpose(kpts_17)
        assert len(result) == 54
        # Nose stays at index 0
        assert result[0] == 0.0  # x
        assert result[1] == 5.0  # y
        # Neck inserted at index 1 (average of RShoulder idx1 and LShoulder idx5)
        # COCO-17: idx 1 = RShoulder (10, 15), idx 5 = LShoulder (50, 55)
        neck_x = (10.0 + 50.0) / 2  # 30.0
        neck_y = (15.0 + 55.0) / 2  # 35.0
        assert result[3] == pytest.approx(neck_x)
        assert result[4] == pytest.approx(neck_y)

    def test_coco18_passthrough(self):
        """Test COCO-18 (54 floats) passes through unchanged."""
        kpts_18 = [float(i) for i in range(54)]
        result, _, _ = extract_keypoints_from_dwpose(kpts_18)
        assert result == [float(i) for i in range(54)]

    def test_multi_person_selection(self):
        """Test selecting different people from multi-person detection."""
        kpts_a = [100.0] * 54
        kpts_b = [200.0] * 54
        pose_data = {
            "people": [
                {"pose_keypoints_2d": kpts_a},
                {"pose_keypoints_2d": kpts_b},
            ],
            "canvas_height": 1024,
            "canvas_width": 768,
        }
        result_a, _, _ = extract_keypoints_from_dwpose(pose_data, person_index=0)
        result_b, _, _ = extract_keypoints_from_dwpose(pose_data, person_index=1)
        assert result_a[0] == 100.0
        assert result_b[0] == 200.0

    def test_empty_input(self):
        result, h, w = extract_keypoints_from_dwpose([])
        assert result == []

    def test_single_person_dict(self):
        kpts = [float(i) for i in range(54)]
        pose_data = {"pose_keypoints_2d": kpts}
        result, _, _ = extract_keypoints_from_dwpose(pose_data)
        assert len(result) == 54

    def test_list_of_person_dicts(self):
        kpts = [float(i) for i in range(54)]
        pose_data = [{"pose_keypoints_2d": kpts}]
        result, _, _ = extract_keypoints_from_dwpose(pose_data)
        assert len(result) == 54
